#!/bin/bash
#SBATCH --job-name=single_hosp_ft
#SBATCH --output=logs/single_hosp_ft_%j.out
#SBATCH --error=logs/single_hosp_ft_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --time=2:00:00

# ============================================================================
# SINGLE-HOSPITAL FINE-TUNING - SLURM WRAPPER
# ============================================================================
# Fine-tunes a fresh CarinaNet on each hospital independently (no weight transfer).
# Uses the exact same CL methodology: find_epoch() -> train -> infer.
#
# Hospitals are split into 3 groups for parallel execution:
#   Group 1 (11): Alberta, American-University-of-Beirut, ... Kirikkale-Hospital-
#   Group 2  (9): Sidra-Health, Tel-Aviv-Medical-Center, ... University-of-Linz
#   Group 3 (10): La-Paz-University-Hospital, ... Shiraz-University
#
# Usage:
#   sbatch run_single_hospital_ft.sh 1          # Group 1, 10 sims (default)
#   sbatch run_single_hospital_ft.sh 2          # Group 2
#   sbatch run_single_hospital_ft.sh 3          # Group 3
#   sbatch run_single_hospital_ft.sh 1 5        # Group 1, 5 sims
#   sbatch run_single_hospital_ft.sh 1 1 --test-mode   # Group 1, test mode
# ============================================================================

# Parse arguments
HOSPITAL_GROUP="${1:?ERROR: Must specify hospital group (1, 2, or 3). Usage: sbatch run_single_hospital_ft.sh <group> [num_sims] [extra_flags]}"
NUM_SIMS="${2:-10}"

# Shift past group and num_sims to capture any extra flags (e.g. --test-mode)
shift  # Remove group (always present)
if [ $# -gt 0 ]; then shift; fi  # Remove num_sims if provided
EXTRA_FLAGS="$@"

echo "========================================"
echo "Single-Hospital Fine-Tuning - Job ${SLURM_JOB_ID}"
echo "Hospital Group: ${HOSPITAL_GROUP}"
echo "Simulations: ${NUM_SIMS}"
echo "Started: $(date)"
echo "========================================"

# Resolve script paths relative to the script's location (works from any cwd).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Paths (override CONFIG_BASE / OUTPUT_BASE here if you want a different setup)
WORKING_DIR="$REPO_DIR"
CONFIG_BASE="${WORKING_DIR}/configs/continual_learning/config_naive.yaml"
OUTPUT_BASE="${WORKING_DIR}/outputs/outputs_single_hospital_ft"

# Create directories
mkdir -p "${WORKING_DIR}/logs"
mkdir -p "${OUTPUT_BASE}"
mkdir -p "${WORKING_DIR}/configs/single_hospital_ft"

# Activate conda
eval "$(conda shell.bash hook)"
conda activate cl

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment"
    exit 1
fi

# Change to repo root so utils.* imports resolve correctly.
cd "$WORKING_DIR"

if [ ! -f "scripts/global_single_hospital_ft.py" ]; then
    echo "ERROR: scripts/global_single_hospital_ft.py not found in ${WORKING_DIR}"
    exit 1
fi

# Create config with output_path pointing to single-hospital FT output
CONFIG_FILE="${WORKING_DIR}/configs/single_hospital_ft/config_ft_group${HOSPITAL_GROUP}_${SLURM_JOB_ID:-local}.yaml"

python -c "
import yaml

with open('${CONFIG_BASE}', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

config['wandb_off'] = True
config['eval_current_hospital_only'] = True
config['output_path'] = '${OUTPUT_BASE}'

with open('${CONFIG_FILE}', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print('Config created: ${CONFIG_FILE}')
"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create config"
    exit 1
fi

echo ""
echo "========================================"
echo "Config: ${CONFIG_FILE}"
echo "Hospital group: ${HOSPITAL_GROUP}"
echo "Simulations: ${NUM_SIMS}"
echo "Extra flags: ${EXTRA_FLAGS}"
echo "Output base: ${OUTPUT_BASE}"
echo "========================================"
echo ""

# Run single-hospital fine-tuning
python scripts/global_single_hospital_ft.py \
    -c "$CONFIG_FILE" \
    --hospital-group "$HOSPITAL_GROUP" \
    --num-sims "$NUM_SIMS" \
    $EXTRA_FLAGS

EXIT_CODE=$?

# Cleanup config
rm -f "$CONFIG_FILE"

echo ""
echo "========================================"
echo "Finished: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "========================================"

exit $EXIT_CODE
