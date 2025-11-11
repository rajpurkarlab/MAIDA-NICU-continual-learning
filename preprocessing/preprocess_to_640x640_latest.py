#!/usr/bin/env python3
"""
Preprocess images and annotations to 640x640 format (Latest Annotations - Valid Only).
- Applies histogram equalization
- Resizes to fit 640 on larger dimension
- Adds padding to make square
- Converts grayscale to RGB (3-channel)
- Updates annotations to new coordinate space
"""

import json
import os
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from tqdm import tqdm
import glob


# ==============================================================================
# CONFIGURATION - UPDATE THESE PATHS FOR YOUR SETUP
# ==============================================================================

# Input: COCO format annotations from Step 1 (convert_to_coco_latest.py output)
ANNOTATIONS_DIR = "/path/to/coco_annotations"  # UPDATE THIS PATH (from Step 1 OUTPUT_DIR)

# Input: Raw DICOM/PNG images organized by hospital (same as Step 1)
IMAGES_DIR = "/path/to/your/raw_images"  # UPDATE THIS PATH

# Output: Preprocessed 640x640 annotations (will be created)
OUTPUT_ANNOTATIONS_DIR = "/path/to/output/annotations_640x640"  # UPDATE THIS PATH

# Output: Preprocessed 640x640 RGB images (will be created)
OUTPUT_IMAGES_DIR = "/path/to/output/images_640x640"  # UPDATE THIS PATH


# ==============================================================================
# PREPROCESSING PARAMETERS
# ==============================================================================

TARGET_SIZE = 640

def apply_histogram_equalization(img_array):
    """Apply histogram equalization to grayscale image."""
    # Convert to uint8 if needed
    if img_array.dtype != np.uint8:
        # Normalize to 0-255
        img_min = img_array.min()
        img_max = img_array.max()
        if img_max > img_min:
            img_array = ((img_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img_array = np.zeros_like(img_array, dtype=np.uint8)

    # Apply histogram equalization
    equalized = cv2.equalizeHist(img_array)
    return equalized

def preprocess_image(image_path, output_path):
    """
    Preprocess a single image:
    1. Load image (handle both grayscale and RGB)
    2. Convert to grayscale if needed
    3. Apply histogram equalization
    4. Resize to fit 640 on larger dimension
    5. Add padding to make 640x640
    6. Convert to RGB (3-channel) for model compatibility

    Returns: scale_factor, padding_info
    """
    # Load image - try grayscale first
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        # Try loading as color and convert
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    orig_height, orig_width = img.shape[:2]

    # Apply histogram equalization BEFORE resizing
    img_equalized = apply_histogram_equalization(img)

    # Calculate scale factor
    scale_factor = TARGET_SIZE / max(orig_width, orig_height)

    # Calculate new dimensions
    new_width = int(orig_width * scale_factor)
    new_height = int(orig_height * scale_factor)

    # Resize using cubic interpolation
    img_resized = cv2.resize(img_equalized, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # Calculate padding
    pad_left = (TARGET_SIZE - new_width) // 2
    pad_right = TARGET_SIZE - new_width - pad_left
    pad_top = (TARGET_SIZE - new_height) // 2
    pad_bottom = TARGET_SIZE - new_height - pad_top

    # Add padding (black = 0)
    img_padded = cv2.copyMakeBorder(
        img_resized,
        pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=0
    )

    # IMPORTANT: Convert grayscale to RGB by duplicating channels
    # The model expects 3-channel input even for X-ray images
    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_GRAY2RGB)

    # Save image as RGB PNG
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    success = cv2.imwrite(output_path, img_rgb)

    if not success:
        raise ValueError(f"Failed to save image: {output_path}")

    # Verify the saved image is RGB
    verify_img = cv2.imread(output_path)
    if verify_img is None or len(verify_img.shape) != 3 or verify_img.shape[2] != 3:
        raise ValueError(f"Saved image is not RGB: {output_path}")

    padding_info = {
        'pad_left': pad_left,
        'pad_right': pad_right,
        'pad_top': pad_top,
        'pad_bottom': pad_bottom
    }

    return scale_factor, padding_info

def transform_bbox(bbox, scale_factor, padding_info):
    """Transform bounding box coordinates to 640x640 space."""
    x, y, w, h = bbox

    # Apply scale
    x_scaled = x * scale_factor
    y_scaled = y * scale_factor
    w_scaled = w * scale_factor
    h_scaled = h * scale_factor

    # Apply padding offset
    x_final = x_scaled + padding_info['pad_left']
    y_final = y_scaled + padding_info['pad_top']

    return [x_final, y_final, w_scaled, h_scaled]

def process_annotations_file(json_file):
    """Process a single annotation file and its images."""
    print(f"\nProcessing {os.path.basename(json_file)}...")

    # Load annotations
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Track processed images to avoid duplicates
    processed_images = {}
    failed_images = []

    # Process each image
    for img_info in tqdm(data['images'], desc="Processing images"):
        file_name = img_info['file_name']
        image_path = os.path.join(IMAGES_DIR, file_name)
        output_image_path = os.path.join(OUTPUT_IMAGES_DIR, file_name)

        # Skip if already processed
        if file_name in processed_images:
            scale_factor = processed_images[file_name]['scale_factor']
            padding_info = processed_images[file_name]['padding_info']
        else:
            try:
                # Preprocess image
                scale_factor, padding_info = preprocess_image(image_path, output_image_path)
                processed_images[file_name] = {
                    'scale_factor': scale_factor,
                    'padding_info': padding_info
                }
            except Exception as e:
                print(f"  Warning: Failed to process {file_name}: {e}")
                failed_images.append(file_name)
                continue

        # Update image info
        img_info['scale_factor'] = scale_factor
        img_info['padding_info'] = padding_info

        # Update pixel spacing
        if 'row_pixel_spacing' in img_info:
            img_info['original_row_pixel_spacing'] = img_info['row_pixel_spacing']
            img_info['row_pixel_spacing'] = img_info['row_pixel_spacing'] / scale_factor
        if 'col_pixel_spacing' in img_info:
            img_info['original_col_pixel_spacing'] = img_info['col_pixel_spacing']
            img_info['col_pixel_spacing'] = img_info['col_pixel_spacing'] / scale_factor

    # Transform annotations
    for ann in data['annotations']:
        # Get corresponding image info
        img_id = ann['image_id']
        img_info = next((img for img in data['images'] if img['id'] == img_id), None)

        if img_info and 'scale_factor' in img_info:
            scale_factor = img_info['scale_factor']
            padding_info = img_info['padding_info']

            # Transform bbox to 640x640 space
            if 'bbox' in ann:
                ann['bbox_original'] = ann['bbox'].copy()  # Keep original
                ann['bbox'] = transform_bbox(ann['bbox'], scale_factor, padding_info)

            # Keep normalized coordinates as they are (they're already normalized)
            # bbox_normalized stays the same

    # Remove failed images and their annotations
    if failed_images:
        # Get IDs of failed images
        failed_ids = set()
        data['images'] = [img for img in data['images']
                         if img['file_name'] not in failed_images or failed_ids.add(img['id'])]

        # Remove annotations for failed images
        data['annotations'] = [ann for ann in data['annotations']
                              if ann['image_id'] not in failed_ids]

        print(f"  Removed {len(failed_images)} failed images and their annotations")

    # Save updated annotations
    output_file = os.path.join(OUTPUT_ANNOTATIONS_DIR, os.path.basename(json_file))
    os.makedirs(OUTPUT_ANNOTATIONS_DIR, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"  Processed {len(processed_images)} images")
    print(f"  Saved to {output_file}")

    return len(processed_images), len(failed_images)

def verify_rgb_conversion():
    """Verify that all output images are RGB."""
    print("\nVerifying RGB conversion...")
    output_images = glob.glob(os.path.join(OUTPUT_IMAGES_DIR, "*.png"))

    if not output_images:
        print("  No images found to verify!")
        return

    grayscale_count = 0
    rgb_count = 0
    error_count = 0

    for img_path in output_images[:10]:  # Check first 10 as sample
        try:
            img = cv2.imread(img_path)
            if img is None:
                error_count += 1
            elif len(img.shape) == 2:
                grayscale_count += 1
            elif len(img.shape) == 3 and img.shape[2] == 3:
                rgb_count += 1
            else:
                error_count += 1
        except Exception as e:
            error_count += 1

    print(f"  Sample verification (first 10 images):")
    print(f"    RGB (3-channel): {rgb_count}")
    print(f"    Grayscale: {grayscale_count}")
    print(f"    Errors: {error_count}")

    if grayscale_count > 0:
        print("  WARNING: Some images are still grayscale!")
    elif rgb_count == 10:
        print("   All sampled images are RGB")

def main():
    # Create output directories
    os.makedirs(OUTPUT_ANNOTATIONS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)

    # Get all annotation files
    annotation_files = glob.glob(os.path.join(ANNOTATIONS_DIR, "*.json"))

    if not annotation_files:
        print(f"No annotation files found in {ANNOTATIONS_DIR}")
        return

    print(f"Found {len(annotation_files)} annotation files to process")

    total_processed = 0
    total_failed = 0

    # Process each annotation file
    for json_file in sorted(annotation_files):
        processed, failed = process_annotations_file(json_file)
        total_processed += processed
        total_failed += failed

    # Verify RGB conversion
    verify_rgb_conversion()

    print(f"\n{'='*80}")
    print(f"PREPROCESSING COMPLETE:")
    print(f"  Total images processed: {total_processed}")
    print(f"  Total images failed: {total_failed}")
    print(f"  Output annotations: {OUTPUT_ANNOTATIONS_DIR}")
    print(f"  Output images: {OUTPUT_IMAGES_DIR}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
