# Data Directory

This directory contains the annotations, clinical metadata, and placeholders for image data required to run the continual learning experiments.

## Included Files

The following files and directories are already included in this repository:

- **`clinical_annotations.json`**: Clinical annotations with ETT placement labels (Low/Normal/High) for each image
- **`annotations/`**: COCO format annotations for all 32 hospitals (both original and preprocessed 640x640)
  - `annotations/original/` - Original resolution annotations
  - `annotations/preprocessed_640x640/` - Preprocessed annotations (used by training scripts)

## Required Downloads

**IMPORTANT**: Due to patient privacy, the actual medical images and patient demographics are not included in this repository. Please replace the placeholder folders with files of your own and annotations in the expected format.

### Directory Structure

After downloading the data, your `data/` directory should have the following structure:

```
data/
├── README.md                           # This file
├── clinical_annotations.json           # ✓ Included
│
├── images/                             # ⚠ DOWNLOAD REQUIRED
│   ├── original/
│   │   └── hospitals/                  # Original resolution images (32 hospitals)
│   │       ├── hospital_image_001.png
│   │       ├── hospital_image_002.png
│   │       └── ...
│   └── preprocessed_640x640/
│       └── hospitals/                  # Preprocessed 640x640 images (32 hospitals)
│           ├── hospital_image_001.png
│           ├── hospital_image_002.png
│           └── ...
│
├── annotations/                        # ✓ Included
│   ├── original/                       # Original resolution annotations (COCO format)
│   │   ├── Alberta-train-annotations.json
│   │   ├── Alberta-test-annotations.json
│   │   ├── American-University-of-Beirut-train-annotations.json
│   │   ├── American-University-of-Beirut-test-annotations.json
│   │   └── ... (32 hospitals × 2 splits = 64 files)
│   │
│   └── preprocessed_640x640/           # 640x640 preprocessed annotations (COCO format)
│       ├── Alberta-train-annotations.json
│       ├── Alberta-test-annotations.json
│       ├── ... (32 hospitals × 2 splits = 64 files)
│       ├── hospital-train-annotations.json  # Combined train set (all hospitals)
│       └── hospital-test-annotations.json   # Combined test set (all hospitals)
```

## Preprocessing Notes

If you need to preprocess your own images or have raw data:
- See `preprocessing/README.md` for instructions on converting raw images to 640x640 format
- See `preprocessing/preprocess_to_640x640_latest.py` for the preprocessing script

For most users, the preprocessed data download is sufficient and no additional preprocessing is needed.

## Questions?

If you encounter issues with data download or placement, please open an issue on the GitHub repository.
