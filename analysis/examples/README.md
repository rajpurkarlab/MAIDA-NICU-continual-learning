# Analysis Examples

This directory contains example scripts demonstrating how to analyze continual learning experiment results.

## Configuration Required

Before running these scripts, you must configure paths to your experiment outputs. The scripts use relative paths for repository data (annotations, demographics) but require you to specify where your experiment results are stored.

### Step 1: Configure Experiment Output Paths

Edit `analysis/utils/prediction_loading_utils.py` and update the following paths (lines 40-91):

```python
# ==============================================================================
# CONFIGURATION - MODIFY THESE PATHS FOR YOUR SETUP
# ==============================================================================

# Continual Learning Output Locations
CL_LOCATIONS = {
    'LOCATION_1': {
        'path': '/path/to/your/continual_learning_outputs/location_1',  # UPDATE THIS PATH
        'hospitals': ['Alberta', 'American-University-of-Beirut', ...]
    },
    'LOCATION_2': {
        'path': '/path/to/your/continual_learning_outputs/location_2',  # UPDATE THIS PATH
        'hospitals': ['Sichuan-People-Hospital', 'Sidra-Health', ...]
    },
    'LOCATION_3': {
        'path': '/path/to/your/continual_learning_outputs/location_3',  # UPDATE THIS PATH
        'hospitals': ['La-Paz-University-Hospital', ...]
    }
}

# Fine-tuning outputs directory (contains simulation_0 through simulation_9 subdirectories)
DEFAULT_FT_DIR = Path("/path/to/your/fine_tuning_outputs")  # UPDATE THIS PATH

# Direct prediction results file
DEFAULT_DIRECT_PATH = Path("/path/to/your/direct_prediction/nicu_test_predictions.csv")  # UPDATE THIS PATH
```

**Important Notes:**
- CL outputs are organized by hospital: `{path}/{hospital}_holdout/simulation_N/predictions.csv`
- FT outputs are organized by simulation: `{path}/simulation_N/finetuned-{hospital}.csv`
- Direct prediction is a single CSV file with all predictions

### Step 2: Configure Output Directories for Generated Figures

#### For create_localization_comparison.py

Edit line 57 in `create_localization_comparison.py`:

```python
# Output directory for generated figures and tables
OUTPUT_DIR = Path("/path/to/output/etable4")  # UPDATE THIS PATH
```

#### For create_clinical_precision_recall.py

Edit line 89 in `create_clinical_precision_recall.py`:

```python
# Default output directory for generated figures and tables
DEFAULT_OUTPUT_DIR = Path("/path/to/output/etable5")  # UPDATE THIS PATH
```

Or use the `--output-dir` command line argument when running the script.

### What You DON'T Need to Change

The following paths are automatically found using relative paths from the repository:
- Annotations (in `data/annotations/`)
- Demographics CSVs (in `data/demographics/`)
- Mappings file (in `data/mappings.csv`)
- Clinical annotations (in `data/clinical_annotations.json`)

## Scripts

### 1. create_localization_comparison.py

Compares localization performance across three methods:
- **Continual Learning (CL)**: Sequential training across hospitals
- **Fine-Tuning (FT)**: Independent training on each hospital
- **Direct Prediction**: Pretrained model without adaptation

**Usage:**
```bash
# After configuring paths (see Configuration section above)
cd analysis/examples
python create_localization_comparison.py

# Optional: run quick sanity check with only 1 simulation
python create_localization_comparison.py --sanity-check
```

**Outputs:**
- `etable4_cl_vs_ft_comparison_ci95.png`: Two-method comparison plot
- `etable4_three_method_comparison_ci95.png`: Three-method comparison plot
- `eTable4_localization_errors_with_stats.csv`: Statistical comparison table
- `eTable4_localization_errors_with_stats.txt`: Human-readable table

**Metrics:**
- Localization error statistics (mean with 95% CI across simulations)
- Statistical comparisons using t-tests
- Hospital-level and aggregate performance

### 2. create_clinical_precision_recall.py

Analyzes clinical performance for "too deep" ETT placement detection:
- Precision@K and Recall@K metrics
- Multi-simulation analysis with error bars
- Flag-based visualizations by country

**Usage:**
```bash
# After configuring paths (see Configuration section above)
cd analysis/examples
python create_clinical_precision_recall.py

# Optional: specify custom output directory
python create_clinical_precision_recall.py --output-dir /path/to/output
```

**Outputs:**
- Precision/recall tables with statistical comparisons
- Improvement visualizations (CL vs FT, CL vs Direct)
- Per-hospital and aggregate metrics
- Flag-annotated plots for international comparison

**Metrics:**
- Precision@K and Recall@K for K=[5, 15]
- Statistical significance testing (t-tests)
- Multi-simulation aggregation with error bars

## Requirements

### Python Dependencies
See `environment.yml` in the repository root for full dependencies. Key packages:
- pandas, numpy
- scipy (for statistical tests)
- matplotlib (for visualizations)
- requests (for downloading country flags)

### Utilities
Both scripts use utilities from `analysis/utils/`:
- `prediction_loading_utils.py`: Load predictions from different experiment types
- `ett_width_utils.py`: ETT width normalization and distance calculations

## Expected Data Structure

### Continual Learning Outputs
CL experiments should be organized by hospital with holdout structure:
```
/path/to/CL/outputs/
└── {hospital}_holdout/
    ├── simulation_0/
    │   └── predictions.csv
    ├── simulation_1/
    │   └── predictions.csv
    ...
    └── simulation_9/
        └── predictions.csv
```

### Fine-Tuning Outputs
FT experiments should be organized by simulation:
```
/path/to/FT/outputs/
├── simulation_0/
│   ├── finetuned-Alberta.csv
│   ├── finetuned-American-University-of-Beirut.csv
│   └── ...
├── simulation_1/
│   └── ...
...
└── simulation_9/
    └── ...
```

### Direct Prediction Outputs
Single file with predictions from pretrained model:
```
/path/to/direct/outputs/
└── nicu_test_predictions.csv
```

### Prediction CSV Format
All prediction files should contain:
- `image_id`: Image identifier
- `tip_x`, `tip_y`: Predicted ETT tip coordinates
- `carina_x`, `carina_y`: Predicted carina coordinates
- `hospital_name`: Hospital identifier

## Troubleshooting

### Import Errors
If you get `ModuleNotFoundError` for utility modules:
- Ensure you're running from the repository root or `analysis/examples/` directory
- The scripts automatically add `analysis/utils/` to the Python path

### Path Errors
If you get `FileNotFoundError`:
- Double-check that you've updated all paths in the Configuration section
- Verify your experiment output directory structure matches the expected format above
- Ensure repository data files exist in `data/` directory

### Missing Functions Error (create_clinical_precision_recall.py)
The script may reference helper functions from `create_etable4_forward_compatible.py` which is not included. If you encounter errors related to missing functions, you may need to:
- Comment out or modify sections using those functions
- Implement the missing helper functions based on your needs

## Notes

- Scripts preserve flag emojis in visualizations for country identification
- Multiple simulations (10 by default) are aggregated using mean ± standard deviation
- Statistical tests use appropriate methods:
  - Two-sample t-test for CL vs FT (paired comparisons across simulations)
  - One-sample t-test for CL vs Direct (CL distribution vs Direct point estimate)
- 95% confidence intervals are computed for all multi-simulation metrics
