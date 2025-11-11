#!/bin/bash

# ==============================================================================
# Continual Learning Holdout Analysis Example
# ==============================================================================
# This script demonstrates how to run leave-one-out holdout analysis for a
# single hospital. The model will be trained on all other hospitals and
# tested on the holdout hospital to evaluate generalization.
# ==============================================================================

echo "========================================"
echo "Continual Learning Holdout Analysis"
echo "Started: $(date)"
echo "========================================"

# ==============================================================================
# CONFIGURATION - UPDATE THESE FOR YOUR SETUP
# ==============================================================================

# Specify which hospital to hold out
HOLDOUT_HOSPITAL="Indus"  # Change this to the hospital you want to hold out

# Path to your base config file (relative to scripts directory)
CONFIG_BASE="../configs/continual_learning/config_naive.yaml"

# Output directory (will create subdirectory for holdout hospital)
OUTPUT_BASE="outputs/continual_learning/holdout_analysis"

# ==============================================================================

echo ""
echo "Configuration:"
echo "  Holdout Hospital: ${HOLDOUT_HOSPITAL}"
echo "  Base Config: ${CONFIG_BASE}"
echo "  Output Directory: ${OUTPUT_BASE}/${HOLDOUT_HOSPITAL}_holdout"
echo ""

# Activate conda environment
echo "Activating conda environment 'cl'..."
eval "$(conda shell.bash hook)"
conda activate cl

# Check if conda environment was activated successfully
if [ $? -ne 0 ]; then
    echo "Failed to activate conda environment 'cl'"
    echo "Trying alternative activation method..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate cl
    if [ $? -ne 0 ]; then
        echo "Both activation methods failed"
        echo "Available conda environments:"
        conda env list
        exit 1
    fi
fi

echo "Conda environment 'cl' activated successfully"
echo "Python path: $(which python)"

# Set the working directory to the scripts location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if config file exists
if [ ! -f "$CONFIG_BASE" ]; then
    echo "Error: Configuration file not found at $CONFIG_BASE"
    echo "Please update the CONFIG_BASE path in this script"
    exit 1
fi

# Create temporary config file for this run
CONFIG_FILE="/tmp/config_holdout_${HOLDOUT_HOSPITAL}.yaml"

echo ""
echo "Creating temporary config file..."

python -c "
import yaml

# Load base config
with open('${CONFIG_BASE}', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Update config for holdout analysis
config['wandb_off'] = True  # Disable wandb tracking
config['number_of_simulation'] = 10  # Run 10 simulations (different hospital orders)
config['skip_simulation'] = -1  # Don't skip any simulations
config['eval_current_hospital_only'] = True  # Evaluate only on current hospital

# Set output path to include holdout hospital name
config['output_path'] = '${OUTPUT_BASE}/${HOLDOUT_HOSPITAL}_holdout'

# Save temporary config
with open('${CONFIG_FILE}', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print('Temporary config created at ${CONFIG_FILE}')
"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create config file"
    exit 1
fi

echo ""
echo "Starting holdout analysis..."
echo "  Training on: All hospitals EXCEPT ${HOLDOUT_HOSPITAL}"
echo "  Testing on: ${HOLDOUT_HOSPITAL} only"
echo "  Running 10 simulations with different hospital orders"
echo ""

# Check GPU availability (optional)
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
    echo ""
fi

# Run the holdout analysis
python global_CL_sequential_holdout_v3.py \
    -c "$CONFIG_FILE" \
    --holdout-hospital "$HOLDOUT_HOSPITAL"

EXIT_CODE=$?

# Cleanup temporary config
rm -f "$CONFIG_FILE"

# Report results
echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Holdout analysis completed"
    echo "Results saved to: ${OUTPUT_BASE}/${HOLDOUT_HOSPITAL}_holdout"
    echo ""
    echo "Expected output structure:"
    echo "  ${OUTPUT_BASE}/${HOLDOUT_HOSPITAL}_holdout/"
    echo "    simulation_0/"
    echo "      training_hospital_0.csv"
    echo "      training_hospital_1.csv"
    echo "      ..."
    echo "      holdout_test_results.csv  <- Test results on ${HOLDOUT_HOSPITAL}"
    echo "      holdout_metadata.json"
    echo "    simulation_1/"
    echo "      ..."
    echo "    ..."
    echo "    simulation_9/"
else
    echo "FAILED: Holdout analysis failed (Exit code: ${EXIT_CODE})"
fi
echo "Finished: $(date)"
echo "========================================"

exit $EXIT_CODE
