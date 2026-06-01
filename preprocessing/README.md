# Image Preprocessing Pipeline

This directory contains scripts for preprocessing NICU chest X-ray images to 640x640 format to match the provided annotations.

### Configuration

Edit `preprocess_to_640x640.py` and update the paths (lines 21-35):

```python
# Input: COCO format annotations
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
python preprocess_to_640x640.py
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

## Annotation Format

See `data/example_annotations.json` for a synthetic, correctly-formatted COCO
file you can use as a structural reference when preparing your own annotations.

