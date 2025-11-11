# Image Preprocessing Pipeline

This directory contains scripts for preprocessing NICU chest X-ray images to 640x640 format to match the provided annotations.

## Quick Start

**Most users only need to run one script**: `preprocess_to_640x640_latest.py`

The repository already includes preprocessed annotations in `data/annotations/preprocessed_640x640/`. You just need to preprocess your raw images to match these annotations.

### Configuration

Edit `preprocess_to_640x640_latest.py` and update the paths (lines 21-35):

```python
# Input: COCO format annotations (already provided in repository)
ANNOTATIONS_DIR = "/path/to/coco_annotations"  # Use: data/annotations/preprocessed_640x640/

# Input: Your raw DICOM/PNG images
IMAGES_DIR = "/path/to/your/raw_images"  # UPDATE THIS PATH

# Output: Preprocessed 640x640 annotations (will be created)
OUTPUT_ANNOTATIONS_DIR = "/path/to/output/annotations_640x640"  # UPDATE THIS PATH

# Output: Preprocessed 640x640 RGB images (will be created)
OUTPUT_IMAGES_DIR = "/path/to/output/images_640x640"  # UPDATE THIS PATH
```

### Usage

```bash
cd preprocessing
python preprocess_to_640x640_latest.py
```

## What the Script Does

The preprocessing script performs the following operations on each image:

1. **Histogram Equalization**: Applied to raw grayscale images to improve contrast
2. **Aspect Ratio Preserving Resize**: Larger dimension scaled to 640px
3. **Padding**: Black borders added to make square 640x640
4. **RGB Conversion**: Grayscale duplicated to 3 channels (required for model)
5. **Coordinate Transformation**: Updates bounding boxes to 640x640 space
6. **Pixel Spacing Update**: Adjusts pixel spacing values for new image scale

### Expected Input

- **Annotations**: COCO format JSON files (provided in `data/annotations/preprocessed_640x640/`)
- **Images**: Raw DICOM or PNG images organized by hospital
  - Grayscale images (will be converted to RGB)
  - Any resolution (will be resized to 640x640)

### Expected Output

- **Annotations**: Updated COCO JSON files with transformed coordinates
- **Images**: 640x640 RGB images ready for training
  - All images are 640x640 pixels
  - All images are 3-channel RGB
  - Histogram equalization applied
  - Aspect ratio preserved with padding

## Reference: COCO Conversion Script

The script `convert_to_coco_latest.py` is provided as **reference only** to show how the original annotations were created. Most users do NOT need to run this script since the preprocessed annotations are already included in the repository.

### When to use `convert_to_coco_latest.py`:

- You have new hospitals to add that aren't in the provided annotations
- You want to regenerate annotations from raw Label Studio format
- You need to modify the annotation format or train/test splits

If you do need to regenerate annotations, see the configuration section at the top of `convert_to_coco_latest.py` (lines 21-41).

## Examples: Adding Hospitals Post-Hoc

The `examples/` directory contains scripts for adding hospitals after initial data preparation (like we did with University of Alberta). These are provided as examples:

- `convert_alberta_to_coco.py`: Convert single hospital annotations
- `preprocess_alberta_to_640x640.py`: Preprocess single hospital images
- `merge_alberta_placements.py`: Merge clinical placement labels
- `merge_alberta_to_combined.py`: Merge into combined dataset

Most users won't need these unless adding new hospitals to an existing dataset.

## Image Requirements

### Input Image Format
- **Format**: DICOM (.dcm) or PNG (.png)
- **Channels**: Grayscale (1-channel) or RGB (3-channel)
- **Resolution**: Any (will be resized)
- **Bit Depth**: 8-bit or 16-bit

### Output Image Format
- **Format**: PNG (.png)
- **Size**: 640x640 pixels
- **Channels**: RGB (3-channel)
- **Bit Depth**: 8-bit

## File Structure

```
preprocessing/
├── README.md                           # This file
├── preprocess_to_640x640_latest.py    # Main script: Image preprocessing
├── convert_to_coco_latest.py          # Reference: Annotation generation
├── fix_filenames.py                   # Utility: Filename normalization
└── examples/                          # Optional: Post-hoc hospital addition
    ├── convert_alberta_to_coco.py
    ├── preprocess_alberta_to_640x640.py
    ├── merge_alberta_placements.py
    └── merge_alberta_to_combined.py
```

## Preprocessing Details

### Histogram Equalization
Applied using OpenCV's `equalizeHist` to improve contrast in chest X-rays. This helps the model learn better features from images with varying exposure levels.

### Resize Strategy
- Calculates scale factor: `scale = 640 / max(width, height)`
- Resizes image maintaining aspect ratio
- Adds black padding to shorter dimension to create 640x640 square
- Stores original dimensions and scale factor in metadata

### Coordinate Transformation
Bounding boxes are transformed from original image space to 640x640 space:
- Original coordinates stored as `bbox_original`
- Transformed coordinates stored as `bbox`
- Normalized coordinates preserved as `bbox_normalized`

### RGB Conversion
- Grayscale images: Duplicates single channel to create 3 identical channels
- RGB images: Kept as-is
- Ensures all images are 3-channel for model compatibility

## Verification

After preprocessing, the script will print verification statistics:

```
Processing complete!
  Total images: 3065
  RGB (3-channel): 3065
  Grayscale: 0
  Errors: 0
✓ All images are RGB
✓ All images are 640x640
```

## Troubleshooting

### "Cannot find image file"
- Check that `IMAGES_DIR` points to your raw images directory
- Ensure images are organized by hospital subdirectories
- Verify image filenames match those in annotation files

### "Annotation file not found"
- Check that `ANNOTATIONS_DIR` points to COCO format annotations
- For first-time users: use `data/annotations/preprocessed_640x640/` from repository

### "Invalid image format"
- Ensure images are DICOM (.dcm) or PNG (.png) format
- For DICOM: Install `pydicom` package if not already installed
- For 16-bit images: Script automatically converts to 8-bit

## Notes

- **Train/test split**: Annotations include pre-split train/test sets (stratified by ETT placement)
- **Reproducibility**: Preprocessing is deterministic (no random operations except histogram equalization)
- **Pixel spacing**: Preserved and updated for image scale in metadata
- **Hospital mapping**: Annotations include hospital names for each image
- **Special hospitals**: Essen, Chiang Mai, Osaka use different ID formats (handled automatically)
