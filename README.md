# Repurposing Adult AI Radiograph Interpretation Models for Neonatal Care Through Continual Learning: An International Multi-Center Study

This repository contains code for the study "Repurposing Adult AI Radiograph Interpretation Models for Neonatal Care Through Continual Learning: An International Multi-Center Study". Here, continual learning across multiple hospitals is used to adapt adult endotracheal tube (ETT) placement detection models to interpret neonatal chest X-rays from neonatal intensive care units (NICUs).

## Overview

This system implements continual learning methods to train ETT detection models sequentially across 31 hospitals from 20 countries worldwide as well as compare against conventional single-site fine-tuning and inference directly from the initial adult model.

### Key Features

- **Multi-Hospital Continual Learning**: Sequential training across 31 international hospitals
- **Naive Fine-tuning**: Simple fine-tuning baseline
- **ETT & Carina Detection**: Localization of endotracheal tube tip and carina landmarks
- **Clinical Analysis Tools**: Precision/recall analysis and ETT placement classification
- **Leave-One-Out Validation**: Holdout analysis for generalization testing

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/MAIDA-NICU-continual-learning.git
cd MAIDA-continual-learning
```

### 2. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate cl
```

### 3. Download Pretrained Model Weights

Download the CarinaNet pretrained model from the official repository:
- Repository: [https://github.com/USM-CHU-FGuyon/CarinaNet](https://github.com/USM-CHU-FGuyon/CarinaNet)
- Place the downloaded model as: `models/CarinaNet/model.pt`

See `models/CarinaNet/model_weights.md` for detailed instructions.

## Quick Start

### Run Leave-One-Out Holdout Analysis

```bash
cd scripts
./run_continual_learning.sh
```

This will run leave-one-out holdout analysis (trains on all hospitals except one, tests on holdout hospital). By default, it holds out "Indus" hospital. Edit the script to change the holdout hospital:

```bash
# In run_continual_learning.sh, change line 21:
HOLDOUT_HOSPITAL="Indus"  # Change to any hospital name
```

### Run Basic Continual Learning

To run continual learning across all hospitals (no holdout):

```bash
cd scripts
python global_CL_sequential.py -c ../configs/continual_learning/config_naive.yaml
```

### Configuration

Edit `configs/continual_learning/config_naive.yaml` to adjust:
- Data paths (images and annotations)
- Number of simulations (different hospital orderings)
- Batch size and learning parameters
- Evaluation strategy (current hospital only vs all hospitals)

## Repository Structure

```
├── data/                           # Data directory (see data/README.md for setup)
│   ├── README.md                  # Data download and setup instructions
│   │
│   ├── images/                    # ⚠ Place images here
│   │   ├── original/
│   │   │   └── hospitals/         # Original resolution images (3,065 images)
│   │   └── preprocessed_640x640/
│   │       └── hospitals/         # Preprocessed 640x640 images (3,065 images)
│   │
│   ├── annotations/               # ✓ INCLUDED - COCO format annotations
│   │   ├── original/              # Original resolution annotations (32 hospitals)
│   │   │   ├── {hospital}-train-annotations.json  # Per-hospital train set
│   │   │   ├── {hospital}-test-annotations.json   # Per-hospital test set
│   │   │   ├── all-hospitals-train-annotations.json
│   │   │   └── all-hospitals-test-annotations.json
│   │   └── preprocessed_640x640/  # 640x640 preprocessed annotations (32 hospitals)
│   │       ├── {hospital}-train-annotations.json
│   │       ├── {hospital}-test-annotations.json
│   │       ├── hospital-train-annotations.json  # Combined train set (all hospitals)
│   │       └── hospital-test-annotations.json   # Combined test set (all hospitals)
│   ├── clinical_annotations.json  # ✓ INCLUDED - ETT placement labels (Low/Normal/High)
│
├── preprocessing/                  # Data preprocessing pipeline
│   ├── convert_to_coco_latest.py  # Reference: Convert to COCO format
│   ├── preprocess_to_640x640_latest.py  # Main: Resize images to 640x640
│   ├── fix_filenames.py           # Utility: Normalize filenames
│   └── README.md                  # Preprocessing documentation
│
├── models/                         # Model architectures
│   ├── ETTModel.py                # Main ETT detection wrapper
│   └── CarinaNet/                 # CarinaNet implementation
│       ├── CarinaNetModel.py      # CarinaNet model wrapper
│       ├── model_weights.md       # Instructions for downloading weights
│       └── retinanet/             # RetinaNet backbone
│           ├── model.py           # RetinaNet architecture
│           ├── losses.py          # Loss functions
│           ├── anchors.py         # Anchor generation
│           └── ...
│
├── utils/                          # Core utilities
│   ├── CL_helpers.py              # Continual learning training functions
│   ├── MAIDA_Dataset.py           # PyTorch dataset class
│   ├── training_helper.py         # Cross-validation and training utilities
│   ├── common_helpers.py          # Data loading helpers
│   ├── fine_tune_helpers.py       # Fine-tuning utilities
│   ├── constants.py               # Hospital list and global constants
│   ├── config_helpers.py          # Config file utilities
│   ├── AnnotationLoader.py        # COCO annotation loader
│   └── ...
│
├── scripts/                        # Main execution scripts
│   ├── continual_learning.py      # Core CL implementation
│   ├── global_CL_sequential.py    # Sequential CL across all hospitals
│   ├── global_CL_sequential_holdout_v3.py  # Leave-one-out validation
│   ├── fine_tune_CarinaNet.py     # Fine-tuning baseline (single hospital)
│   ├── inference.py               # Model inference
│   └── run_continual_learning.sh  # Example bash script (holdout analysis)
│
├── analysis/                       # Analysis tools
│   ├── utils/                     # Analysis utility modules
│   │   ├── prediction_loading_utils.py  # Load CL/FT/Direct predictions
│   │   └── ett_width_utils.py     # ETT width normalization & distance calc
│   └── examples/                  # Example analysis scripts
│       ├── localization_comparison.py  # Compare CL vs FT vs Direct
│       ├── clinical_precision_recall.py  # Clinical metrics analysis
│       └── README.md              # Analysis configuration guide
│
├── configs/                        # Configuration files
│   ├── continual_learning/
│   │   └── config_naive.yaml      # Main CL configuration
│   └── fine_tuning/               # Per-hospital fine-tuning configs
│       ├── Alberta.yaml
│       ├── Indus.yaml
│       └── ...  # 32 hospital configs
│
├── environment.yml                 # Conda environment specification
├── .gitignore                      # Git ignore patterns
└── README.md                       # This file
```

## Usage

### Continual Learning

Run naive continual learning across all hospitals:

```bash
cd scripts
python global_CL_sequential.py -c ../configs/continual_learning/config_naive.yaml
```

This trains sequentially on all hospitals with 10 different random orderings (simulations).

### Leave-One-Out Validation

Test generalization by holding out one hospital:

```bash
python global_CL_sequential_holdout_v3.py \
    -c ../configs/continual_learning/config_naive.yaml \
    --holdout-hospital Indus
```

This trains on all hospitals except "Indus" and tests only on "Indus" data.

### Fine-Tuning Baseline

Run single-hospital fine-tuning for comparison:

```bash
python fine_tune_CarinaNet.py -c ../configs/fine_tuning/Indus.yaml
```

Each hospital has its own config file in `configs/fine_tuning/`.

### Analysis

Before running analysis scripts, configure experiment output paths in `analysis/utils/prediction_loading_utils.py` (see `analysis/examples/README.md` for detailed instructions).

Compare continual learning vs fine-tuning vs direct prediction (localization errors):

```bash
cd analysis/examples
python localization_comparison.py
```

Generate clinical precision/recall metrics for ETT placement:

```bash
python clinical_precision_recall.py
```

## Data Format

### Annotations (COCO Format)

```json
{
  "images": [
    {
      "id": "hospital_image_001.png",
      "file_name": "hospital_image_001.png",
      "width": 640,
      "height": 640,
      "pixel_spacing": [0.15, 0.15]
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": "hospital_image_001.png",
      "category_id": 1,
      "bbox": "[x, y, width, height]"
    }
  ],
  "categories": [
    {"id": 1, "name": "tip"},
    {"id": 2, "name": "carina"}
  ]
}
```

## Preprocessing Pipeline

**Step 1**: Preprocess your raw images to 640x640 format:
```bash
cd preprocessing
# Edit preprocess_to_640x640_latest.py to configure paths (lines 21-35)
python preprocess_to_640x640_latest.py
```

This will:
- Apply histogram equalization
- Resize images to 640x640 (preserving aspect ratio with padding)
- Convert grayscale to 3-channel RGB
- Update bounding box coordinates

See `preprocessing/README.md` for detailed instructions and configuration options.

## Citation

If you use this code, please cite:

```
[Citation information to be added]
```

## Acknowledgments

- CarinaNet pretrained model: [https://github.com/USM-CHU-FGuyon/CarinaNet](https://github.com/USM-CHU-FGuyon/CarinaNet)
