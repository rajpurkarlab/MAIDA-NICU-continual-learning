# Data Directory

This directory contains the annotations, clinical metadata, and placeholders for image data required to run the continual learning experiments.

## Included Files

The following files and directories are already included in this repository:

- **`clinical_annotations.json`**: Clinical annotations with ETT placement labels (Low/Normal/High) for each image
- **`mappings.csv`**: Maps random image IDs to hospital names for privacy
- **`annotations/`**: COCO format annotations for all 32 hospitals (both original and preprocessed 640x640)
  - `annotations/original/` - Original resolution annotations
  - `annotations/preprocessed_640x640/` - Preprocessed annotations (used by training scripts)

## Required Downloads

**IMPORTANT**: Due to patient privacy, the actual medical images and patient demographics are not included in this repository. You must download these files separately and place them in the appropriate directories.

### Directory Structure

After downloading the data, your `data/` directory should have the following structure:

```
data/
├── README.md                           # This file
├── clinical_annotations.json           # ✓ Included
├── mappings.csv                        # ✓ Included
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
│
└── demographics/                       # ⚠ DOWNLOAD REQUIRED
    ├── README.md                       # Download instructions
    ├── Alberta_nicu_settings.csv
    ├── American_University_of_Beirut_nicu_settings.csv
    └── ... (32 CSV files, one per hospital)
```

## Download Instructions

### Step 1: Download the Dataset

Download the dataset package containing:
- Medical images (original and preprocessed 640x640)
- Patient demographics CSV files

**[Data download instructions will be provided upon publication]**

### Step 2: Extract and Place Files

After downloading, extract the files and place them in the corresponding directories:

1. **Images**: Place all image files in:
   - `data/images/original/hospitals/` - Original resolution images
   - `data/images/preprocessed_640x640/hospitals/` - Preprocessed 640x640 images

2. **Demographics**: Place all demographics CSV files in:
   - `data/demographics/` - Patient demographics (age, weight, gestational age)

### Step 3: Verify File Placement

After placing the files, verify the structure:

```bash
# From repository root
ls data/images/preprocessed_640x640/hospitals/ | wc -l
# Should show 3065 (total number of images)

ls data/demographics/*.csv | wc -l
# Should show 32 (one per hospital)
```

## Dataset Details

- **Total Images**: 3,065 NICU chest X-rays
- **Hospitals**: 32 hospitals from 21 countries (31 used in main experiments)
- **Image Format**: PNG files (640×640 RGB preprocessed)
- **Annotation Format**: COCO format JSON with bounding boxes for ETT tip and carina
- **Train/Test Split**: ~50 training and ~50 test images per hospital
- **Demographics**: CSV files with patient age, weight, gestational age per image

## Privacy and Ethics

All images have been de-identified and anonymized. Random IDs are used instead of patient identifiers. The `mappings.csv` file maps these random IDs to hospital names only.

## Preprocessing Notes

If you need to preprocess your own images or have raw data:
- See `preprocessing/README.md` for instructions on converting raw images to 640x640 format
- See `preprocessing/preprocess_to_640x640_latest.py` for the preprocessing script

For most users, the preprocessed data download is sufficient and no additional preprocessing is needed.

## Questions?

If you encounter issues with data download or placement, please open an issue on the GitHub repository.
