#!/usr/bin/env python3
"""
Demo Script: ETT Detection on NICU Chest X-rays

This script performs inference on new chest X-ray images using the pre-trained
continual learning model. It uses the exact same preprocessing and inference
functions as the main training pipeline.

Usage:
    python demo/run_demo.py
    python demo/run_demo.py --input-dir /path/to/images --output-dir /path/to/output
    python demo/run_demo.py --model-path /path/to/model.pth
"""

import os
import sys

# Make the repository root importable so `utils`, `models`, and sibling
# scripts resolve when this file is run directly (python <path>/script.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path

# Add repository root to path for imports
REPO_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(REPO_ROOT))

# Add the CarinaNet directory to path so 'retinanet' module can be found
# The saved model references 'retinanet' directly, not 'models.CarinaNet.retinanet'
CARINANET_DIR = REPO_ROOT / "models" / "CarinaNet"
sys.path.insert(0, str(CARINANET_DIR))

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms

# Import preprocessing functions from the existing preprocessing module
from preprocessing.preprocess_to_640x640 import (
    apply_histogram_equalization,
    TARGET_SIZE
)

# Import model and constants from the existing modules
from models.CarinaNet.CarinaNetModel import CarinaNetModel
from utils.constants import (
    COCO_LABELS,
    ANNO_CAT_TIP,
    ANNO_CAT_CARINA,
    NAIVE_UPDATE,
    CUDA_AVAILABLE,
)

# Default paths (relative to demo directory)
DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "CL_pretrained.pth"
DEFAULT_INPUT_DIR = Path(__file__).parent / "input_images"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "output"

# Supported image extensions (PNG only)
SUPPORTED_EXTENSIONS = {'.png'}


def preprocess_image_for_inference(image_path: str, output_path: str = None):
    """
    Preprocess a single image for model inference.

    Uses the exact same preprocessing pipeline as the training data:
    1. Load image (handle grayscale and RGB)
    2. Apply histogram equalization
    3. Resize to fit 640 on larger dimension
    4. Add padding to make 640x640
    5. Convert to RGB (3-channel)

    Args:
        image_path: Path to input image
        output_path: Optional path to save preprocessed image

    Returns:
        tuple: (preprocessed_tensor, original_width, original_height, scale_factor, padding_info)
    """
    # Load image - try grayscale first (same as preprocessing script)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        # Try loading as color and convert
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    orig_height, orig_width = img.shape[:2]

    # Apply histogram equalization BEFORE resizing (same as preprocessing script)
    img_equalized = apply_histogram_equalization(img)

    # Calculate scale factor
    scale_factor = TARGET_SIZE / max(orig_width, orig_height)

    # Calculate new dimensions
    new_width = int(orig_width * scale_factor)
    new_height = int(orig_height * scale_factor)

    # Resize using cubic interpolation (same as preprocessing script)
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

    # Convert grayscale to RGB by duplicating channels (same as preprocessing script)
    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_GRAY2RGB)

    # Save preprocessed image if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img_rgb)

    # Convert to tensor for model input (same as MAIDA_Dataset)
    # PIL expects RGB, cv2 gives BGR, so convert
    img_pil = Image.fromarray(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
    import torchvision.transforms as transforms
    transform = transforms.ToTensor()
    img_tensor = transform(img_pil)

    padding_info = {
        'pad_left': pad_left,
        'pad_right': pad_right,
        'pad_top': pad_top,
        'pad_bottom': pad_bottom
    }

    return img_tensor, orig_width, orig_height, scale_factor, padding_info


def load_model(model_path: str):
    """
    Load the continual learning pre-trained model using the EXACT same
    approach as CarinaNetModel.__init__.

    Args:
        model_path: Path to the CL_pretrained.pth model file

    Returns:
        CarinaNetModel instance
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Please download CL_pretrained.pth and place it in demo/models/"
        )

    print(f"  CUDA available: {CUDA_AVAILABLE}")
    if CUDA_AVAILABLE:
        torch.cuda.empty_cache()
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Load checkpoint - this is the saved DataParallel model
    checkpoint = torch.load(model_path, weights_only=False)

    # Use CarinaNetModel constructor with the checkpoint
    # This is the EXACT same code path used in the training pipeline
    model = CarinaNetModel(
        model_path="",  # Empty string means use default, but we provide checkpoint
        update_method=NAIVE_UPDATE,
        checkpoint=checkpoint,
        use_random_init=False
    )

    return model


def run_inference_on_image(model, image_tensor, image_id):
    """
    Run inference on a single preprocessed image.

    Uses the exact same prediction method as the training pipeline.

    Args:
        model: CarinaNetModel instance
        image_tensor: Preprocessed image tensor (3, 640, 640)
        image_id: Identifier for the image

    Returns:
        dict: Predictions with tip and carina detections
    """
    # Use the model's predict method (same as CL pipeline)
    predictions = model.predict(zip([image_tensor], [image_id]))
    return predictions.get(image_id, {})


def format_predictions_to_dataframe(all_predictions: list) -> pd.DataFrame:
    """
    Format predictions to a DataFrame matching the CL output format.

    Args:
        all_predictions: List of prediction dictionaries

    Returns:
        pd.DataFrame with formatted predictions
    """
    rows = []
    for pred in all_predictions:
        for category in ['tip', 'carina']:
            if category in pred:
                rows.append({
                    'file_name': pred['file_name'],
                    'category': category,
                    'pred_x': pred[category]['pred'][0],
                    'pred_y': pred[category]['pred'][1],
                    'confidence': pred[category]['confidence'],
                    'image_width': pred.get('original_width'),
                    'image_height': pred.get('original_height'),
                    'scale_factor': pred.get('scale_factor'),
                })

    return pd.DataFrame(rows)


def find_images(input_dir: str) -> list:
    """
    Find all supported image files in the input directory.

    Args:
        input_dir: Directory to search for images

    Returns:
        List of image file paths
    """
    image_files = []
    input_path = Path(input_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    for ext in SUPPORTED_EXTENSIONS:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))

    # Filter out hidden files and .gitkeep
    image_files = [f for f in image_files if not f.name.startswith('.')]

    return sorted(image_files)


def main():
    parser = argparse.ArgumentParser(
        description='Run ETT detection demo on NICU chest X-rays'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help='Path to the pre-trained model (default: demo/models/CL_pretrained.pth)'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default=str(DEFAULT_INPUT_DIR),
        help='Directory containing input images (default: demo/input_images/)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help='Directory for output files (default: demo/output/)'
    )
    parser.add_argument(
        '--save-preprocessed',
        action='store_true',
        default=True,
        help='Save preprocessed images (default: True)'
    )

    args = parser.parse_args()

    # Print header
    print("=" * 80)
    print("MAIDA-NICU ETT Detection Demo")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print("=" * 80)
    print()

    # Find input images
    print("Scanning for images...")
    try:
        image_files = find_images(args.input_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not image_files:
        print(f"No images found in {args.input_dir}")
        print(f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}")
        sys.exit(1)

    print(f"Found {len(image_files)} image(s)")
    print()

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_dir = output_dir / "preprocessed_images"
    if args.save_preprocessed:
        preprocessed_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading model...")
    try:
        model = load_model(args.model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    print()

    # Process images
    print("Processing images...")
    all_predictions = []
    failed_images = []

    for image_path in tqdm(image_files, desc="Inference"):
        file_name = image_path.name

        try:
            # Preprocess image
            preprocessed_path = str(preprocessed_dir / f"{image_path.stem}_preprocessed.png") if args.save_preprocessed else None
            img_tensor, orig_w, orig_h, scale, padding = preprocess_image_for_inference(
                str(image_path),
                preprocessed_path
            )

            # Run inference
            predictions = run_inference_on_image(model, img_tensor, file_name)

            # Store results
            result = {
                'file_name': file_name,
                'original_width': orig_w,
                'original_height': orig_h,
                'scale_factor': scale,
            }

            # Extract tip and carina predictions
            for class_id, class_name in [(1, 'tip'), (0, 'carina')]:
                if class_id in predictions:
                    result[class_name] = predictions[class_id]

            all_predictions.append(result)

            # Print per-image results
            tip_info = f"Tip: ({result.get('tip', {}).get('pred', ['N/A'])[0]:.1f}, {result.get('tip', {}).get('pred', ['N/A', 'N/A'])[1]:.1f})" if 'tip' in result else "Tip: not detected"
            carina_info = f"Carina: ({result.get('carina', {}).get('pred', ['N/A'])[0]:.1f}, {result.get('carina', {}).get('pred', ['N/A', 'N/A'])[1]:.1f})" if 'carina' in result else "Carina: not detected"

        except Exception as e:
            failed_images.append((file_name, str(e)))
            print(f"  Failed to process {file_name}: {e}")

    print()

    # Save predictions to CSV
    if all_predictions:
        df = format_predictions_to_dataframe(all_predictions)
        output_csv = output_dir / "predictions.csv"
        df.to_csv(output_csv, index=False)
        print(f"Results saved to: {output_csv}")

    # Print summary
    print()
    print("=" * 80)
    print("Demo complete!")
    print(f"  Processed: {len(all_predictions)} images")
    if failed_images:
        print(f"  Failed: {len(failed_images)} images")
        for fname, error in failed_images:
            print(f"    - {fname}: {error}")
    print(f"  Results saved to: {output_dir / 'predictions.csv'}")
    if args.save_preprocessed:
        print(f"  Preprocessed images saved to: {preprocessed_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
