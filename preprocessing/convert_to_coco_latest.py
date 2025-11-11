#!/usr/bin/env python3
"""
Convert latest annotation format to COCO format with stratified train/test splits.
Filters out PROBLEM status annotations - only processes valid annotations.
Maintains both normalized and absolute coordinates.
"""

import json
import os
import csv
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
import glob
import unicodedata

# Paths
LATEST_ANNOTATIONS_DIR = "/n/data1/hms/dbmi/rajpurkar/lab/datasets/MAIDA-NICU/preprocessed_vish/new_annotations/latest_annotations"
IMAGES_DIR = "/n/data1/hms/dbmi/rajpurkar/lab/datasets/MAIDA-NICU/images/hospitals"
MAPPINGS_FILE = "/home/vir247/scratch/nicu/MAIDA-continual-learning/mappings.csv"
OUTPUT_DIR = "/n/data1/hms/dbmi/rajpurkar/lab/datasets/MAIDA-NICU/preprocessed_vish/new_annotations/annotations_latest"
EXISTING_ANNOTATIONS_DIR = "/n/data1/hms/dbmi/rajpurkar/lab/datasets/MAIDA-NICU/preprocessed_vish/annotations"

# Category mapping
CATEGORY_MAP = {
    "ET tube tip": 1,
    "Carina": 2,
    "Top of the first thoracic vertebra": 3,
    "Top of the first thoracic vertebra ": 3,  # Note trailing space
    "Bottom of the second thoracic vertebra": 4,
    "Bottom of the second thoracic vertebra ": 4  # Note trailing space
}

CATEGORY_INFO = [
    {"id": 1, "name": "tip", "supercategory": "medical_device"},
    {"id": 2, "name": "carina", "supercategory": "anatomical_landmark"},
    {"id": 3, "name": "top_thoracic_vertebra", "supercategory": "anatomical_landmark"},
    {"id": 4, "name": "bottom_thoracic_vertebra", "supercategory": "anatomical_landmark"}
]

def normalize_filename(name):
    """Normalize filename for comparison."""
    if name.endswith('.png'):
        name = name[:-4]
    name = name.lower()
    name = unicodedata.normalize('NFC', name)
    return name

def load_mappings():
    """Load the mappings from random-id to hospital names."""
    mappings = {}
    with open(MAPPINGS_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Map new_name (random-id) to institution
            if row['new_name'] and row['institution']:
                mappings[row['new_name']] = row['institution']
                # Also add without .png extension
                mappings[row['new_name'].replace('.png', '')] = row['institution']
    return mappings

def get_hospital_name_from_mappings(filename, mappings):
    """Get hospital name from mappings."""
    # Try with and without .png
    base_name = filename.replace('.png', '')
    if filename in mappings:
        return mappings[filename]
    elif base_name in mappings:
        return mappings[base_name]
    return None

def normalize_hospital_name(hospital_name):
    """Normalize hospital name to match existing format."""
    # Map to existing hospital naming conventions
    hospital_map = {
        "Fundación Santa Fe de Bogotá": "Fundacion-Santa-Fe-de-Bogota",
        "American University of Beirut": "American-University-of-Beirut",
        "Children's Hospital Colorado": "Children's-Hospital-Colorado",
        "Chulalongkorn University": "Chulalongkorn-University",
        "Indus Hospital": "Indus",
        "Sardjito Hospital": "Sardjito-Hospital",
        "Dr Sardjito Hospital": "Dr-Sardjito-Hospital",
        "University of Graz": "University-of-Graz",
        "Universitätsklinikum Essen": "Universitaetsklinikum-Essen",
        "Maharaj Nakorn Chiang Mai Hospital": "Maharaj-Nakorn-Chiang-Mai-Hospital",
        "Osaka Metropolitan University": "Osaka-Metropolitan-University",
        "National University (Singapore)": "National-University-Singapore",
        "Uni-Tubingen": "Uni-Tubingen",
        "Shiraz University": "Shiraz-University",
        "Tri-Service General Hospital": "Tri-Service-General-Hospital",
        "SES": "SES",
        "Medical Center of South": "Medical-Center-of-South",
        "La Paz University Hospital": "La-Paz-University-Hospital",
        "Sidra Health": "Sidra-Health",
        "International Islamic Medical University": "International-Islamic-Medical-University"
    }

    # Return mapped name or original with spaces replaced
    if hospital_name in hospital_map:
        return hospital_map[hospital_name]
    return hospital_name.replace(' ', '-')

def find_special_hospital_image(annotation_name, hospital_type):
    """Find the actual image file for special hospitals (Essen, Chiang Mai, Osaka)."""
    # Extract patient number from annotation name
    if hospital_type == "essen":
        # "Universitätsklinikum_Essen_p4" -> "p4"
        parts = annotation_name.split('_p')
        if len(parts) > 1:
            patient_num = 'p' + parts[-1].replace('.png', '')
            pattern = f"{IMAGES_DIR}/essen_{patient_num}-study-generated_id_*.png"
    elif hospital_type == "chiang_mai":
        # "Maharaj_Nakorn_Chiang_Mai_Hospital_p26" -> "p26"
        parts = annotation_name.split('_p')
        if len(parts) > 1:
            patient_num = 'p' + parts[-1].replace('.png', '')
            pattern = f"{IMAGES_DIR}/chiang_mai_{patient_num}-study-generated_id_*.png"
    elif hospital_type == "osaka":
        # "Osaka_Metropolitan_University_p15" -> "p15"
        parts = annotation_name.split('_p')
        if len(parts) > 1:
            patient_num = 'p' + parts[-1].replace('.png', '')
            pattern = f"{IMAGES_DIR}/osaka_{patient_num}-study-generated_id_*.png"
    else:
        return None

    # Find matching file
    matches = glob.glob(pattern)
    if matches:
        return os.path.basename(matches[0])
    return None

def get_pixel_spacing(hospital_name):
    """Get pixel spacing from existing annotations or use default."""
    # Try to load from existing annotations
    existing_file = os.path.join(EXISTING_ANNOTATIONS_DIR, f"{hospital_name}-train-annotations.json")
    if os.path.exists(existing_file):
        try:
            with open(existing_file, 'r') as f:
                data = json.load(f)
                if data.get('images') and len(data['images']) > 0:
                    img = data['images'][0]
                    return img.get('row_pixel_spacing', 0.14), img.get('col_pixel_spacing', 0.14)
        except:
            pass
    return 0.14, 0.14  # Default values

def process_annotation(ann_data, mappings, missing_annotations, problem_count):
    """Process a single annotation entry. Returns None if PROBLEM status."""
    name = ann_data.get('name', '')
    status = ann_data.get('status', 'OK')

    # FILTER: Skip PROBLEM status annotations
    if status == 'PROBLEM':
        problem_count[0] += 1
        return None

    # Determine hospital and file name
    hospital_name = None
    actual_filename = None

    # Check if it's a special hospital
    if 'Essen' in name or 'Universitätsklinikum_Essen' in name:
        hospital_name = "Universitaetsklinikum-Essen"
        actual_filename = find_special_hospital_image(name, "essen")
    elif 'Maharaj_Nakorn_Chiang_Mai' in name:
        hospital_name = "Maharaj-Nakorn-Chiang-Mai-Hospital"
        actual_filename = find_special_hospital_image(name, "chiang_mai")
    elif 'Osaka_Metropolitan' in name:
        hospital_name = "Osaka-Metropolitan-University"
        actual_filename = find_special_hospital_image(name, "osaka")
    else:
        # Regular hospital with random-id
        actual_filename = name if name.endswith('.png') else name + '.png'
        hospital_name = get_hospital_name_from_mappings(actual_filename, mappings)
        if hospital_name:
            hospital_name = normalize_hospital_name(hospital_name)

    if not actual_filename or not hospital_name:
        missing_annotations.append({'filename': name, 'hospital': 'Unknown', 'reason': 'No mapping found'})
        return None

    # Check if image exists
    image_path = os.path.join(IMAGES_DIR, actual_filename)
    if not os.path.exists(image_path):
        missing_annotations.append({'filename': actual_filename, 'hospital': hospital_name, 'reason': 'Image not found'})
        return None

    # Load image to get dimensions
    try:
        img = Image.open(image_path)
        orig_width, orig_height = img.size
        # Verify we got actual dimensions
        if orig_width == 0 or orig_height == 0:
            missing_annotations.append({'filename': actual_filename, 'hospital': hospital_name, 'reason': 'Invalid image dimensions'})
            return None
    except Exception as e:
        missing_annotations.append({'filename': actual_filename, 'hospital': hospital_name, 'reason': f'Cannot load image: {e}'})
        return None

    # Get ETT placement
    ett_placement = None
    if 'classification' in ann_data and 'attributes' in ann_data['classification']:
        if 'ETT placement' in ann_data['classification']['attributes']:
            ett_placement = ann_data['classification']['attributes']['ETT placement'][0]

    # Process bounding boxes
    annotations = []
    if 'series' in ann_data and ann_data['series']:
        series = ann_data['series'][0]
        if 'boundingBoxes' in series:
            for bbox_data in series['boundingBoxes']:
                category_name = bbox_data.get('category', '').strip()
                if category_name not in CATEGORY_MAP:
                    print(f"Warning: Unknown category '{category_name}' in {actual_filename}")
                    continue

                # Get normalized coordinates
                x_norm = bbox_data['pointTopLeft']['xNorm']
                y_norm = bbox_data['pointTopLeft']['yNorm']
                w_norm = bbox_data['wNorm']
                h_norm = bbox_data['hNorm']

                # Convert to absolute coordinates (original image space)
                x_abs = x_norm * orig_width
                y_abs = y_norm * orig_height
                w_abs = w_norm * orig_width
                h_abs = h_norm * orig_height

                annotations.append({
                    'bbox': [x_abs, y_abs, w_abs, h_abs],
                    'bbox_normalized': [x_norm, y_norm, w_norm, h_norm],
                    'category_id': CATEGORY_MAP[category_name],
                    'category_name': category_name.strip(),
                    'area': w_abs * h_abs
                })

    return {
        'filename': actual_filename,
        'hospital_name': hospital_name,
        'orig_width': orig_width,
        'orig_height': orig_height,
        'ett_placement': ett_placement,
        'annotations': annotations
    }

def stratified_split(hospital_data, train_size=50):
    """Create stratified train/test split maintaining ETT placement distribution."""
    # If we don't have enough data for train+test split
    if len(hospital_data) <= train_size:
        # Put all in train if we have 50 or fewer images
        return hospital_data, []

    # Group by ETT placement
    placement_groups = defaultdict(list)
    for item in hospital_data:
        placement = item.get('ett_placement', 'Unknown')
        placement_groups[placement].append(item)

    train_data = []
    test_data = []

    # First pass: proportionally allocate to maintain distribution
    total = len(hospital_data)
    train_ratio = train_size / total

    # Sort groups by size (largest first) to handle allocation better
    sorted_groups = sorted(placement_groups.items(), key=lambda x: len(x[1]), reverse=True)

    allocated = 0
    allocations = []

    for placement, items in sorted_groups:
        # Calculate proportional allocation
        ideal_n_train = int(len(items) * train_ratio)
        # Ensure at least 1 if the group is large enough
        if len(items) > 1 and ideal_n_train == 0:
            ideal_n_train = 1

        # Don't exceed the group size
        ideal_n_train = min(ideal_n_train, len(items))

        allocations.append((placement, items, ideal_n_train))
        allocated += ideal_n_train

    # Second pass: adjust to get exactly train_size
    remaining = train_size - allocated

    # If we need more samples
    if remaining > 0:
        for i, (placement, items, n_train) in enumerate(allocations):
            available = len(items) - n_train
            if available > 0:
                take = min(available, remaining)
                allocations[i] = (placement, items, n_train + take)
                remaining -= take
                if remaining == 0:
                    break

    # If we have too many samples (reduce from smallest groups first)
    elif remaining < 0:
        need_to_remove = -remaining
        # Sort by allocation size (smallest first)
        for i in sorted(range(len(allocations)), key=lambda x: allocations[x][2]):
            placement, items, n_train = allocations[i]
            if n_train > 1:  # Keep at least 1 from each group if possible
                remove = min(n_train - 1, need_to_remove)
                allocations[i] = (placement, items, n_train - remove)
                need_to_remove -= remove
                if need_to_remove == 0:
                    break

        # If still need to remove, remove from groups with allocation of 1
        if need_to_remove > 0:
            for i in range(len(allocations)):
                placement, items, n_train = allocations[i]
                if n_train > 0:
                    remove = min(n_train, need_to_remove)
                    allocations[i] = (placement, items, n_train - remove)
                    need_to_remove -= remove
                    if need_to_remove == 0:
                        break

    # Final pass: do the actual split
    for placement, items, n_train in allocations:
        if n_train > 0 and n_train < len(items):
            # Use stratified split
            indices = np.arange(len(items))
            np.random.seed(42)  # For reproducibility
            np.random.shuffle(indices)
            train_indices = indices[:n_train]
            test_indices = indices[n_train:]

            train_data.extend([items[i] for i in train_indices])
            test_data.extend([items[i] for i in test_indices])
        elif n_train >= len(items):
            # All go to train
            train_data.extend(items)
        else:
            # All go to test (n_train == 0)
            test_data.extend(items)

    # Verify we have exactly train_size
    assert len(train_data) == train_size, f"Train size is {len(train_data)}, expected {train_size}"

    return train_data, test_data

def create_coco_format(data_items, hospital_name, split_type):
    """Create COCO format dictionary from processed data."""
    row_spacing, col_spacing = get_pixel_spacing(hospital_name)

    coco_data = {
        "info": {
            "description": f"NICU Dataset - {hospital_name} {split_type} set (Latest Annotations - Valid Only)",
            "url": "",
            "version": "2.0",
            "year": 2025,
            "contributor": "NICU Research Team",
            "date_created": datetime.now().strftime("%Y-%m-%d")
        },
        "licenses": [
            {"id": 1, "name": "Research Use", "url": ""}
        ],
        "images": [],
        "annotations": [],
        "categories": CATEGORY_INFO
    }

    img_id = 1
    ann_id = 1

    for item in data_items:
        # Add image
        image_info = {
            "id": img_id,
            "orig_width": item['orig_width'],
            "orig_height": item['orig_height'],
            "width": 640,  # Target size (not resized yet)
            "height": 640,
            "file_name": item['filename'],
            "license": 1,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "hospital_name": hospital_name,
            "original_name": item['filename'],
            "row_pixel_spacing": row_spacing,
            "col_pixel_spacing": col_spacing,
            "ett_placement": item.get('ett_placement', None)
        }
        coco_data['images'].append(image_info)

        # Add annotations
        for ann in item['annotations']:
            ann_info = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": ann['category_id'],
                "segmentation": [],
                "bbox": ann['bbox'],
                "bbox_normalized": ann['bbox_normalized'],
                "iscrowd": 0,
                "area": ann['area'],
                "category_name": ann['category_name']
            }
            coco_data['annotations'].append(ann_info)
            ann_id += 1

        img_id += 1

    return coco_data

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load mappings
    print("Loading mappings...")
    mappings = load_mappings()

    # Track statistics
    missing_annotations = []
    problem_count = [0]  # Use list to pass by reference

    # Process all annotations
    hospital_data = defaultdict(list)

    # Load from latest_annotations folder
    annotation_files = [
        ("tasks.json", "Main tasks"),
        ("essen.json", "Essen Hospital"),
        ("chiang_mai.json", "Chiang Mai Hospital"),
        ("osaka.json", "Osaka Hospital")
    ]

    total_processed = 0
    for filename, display_name in annotation_files:
        print(f"\nProcessing {display_name}...")
        file_path = os.path.join(LATEST_ANNOTATIONS_DIR, filename)

        if not os.path.exists(file_path):
            print(f"  WARNING: {file_path} not found!")
            continue

        with open(file_path, 'r') as f:
            data = json.load(f)

        for ann_data in data:
            processed = process_annotation(ann_data, mappings, missing_annotations, problem_count)
            if processed:
                hospital_data[processed['hospital_name']].append(processed)
                total_processed += 1

    print(f"\n{'='*80}")
    print(f"FILTERING SUMMARY:")
    print(f"  Total PROBLEM status (filtered out): {problem_count[0]}")
    print(f"  Total valid annotations processed: {total_processed}")
    print(f"{'='*80}")

    # Save missing annotations
    if missing_annotations:
        missing_file = os.path.join(OUTPUT_DIR, "missing_annotations.csv")
        with open(missing_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['filename', 'hospital', 'reason'])
            writer.writeheader()
            writer.writerows(missing_annotations)
        print(f"Saved {len(missing_annotations)} missing annotations to {missing_file}")

    # Create train/test splits and save for each hospital
    all_train = []
    all_test = []

    for hospital_name, data in sorted(hospital_data.items()):
        print(f"\nProcessing {hospital_name}: {len(data)} images")

        # Create stratified split
        train_data, test_data = stratified_split(data, train_size=50)

        # Print distribution
        train_dist = Counter([d.get('ett_placement', 'Unknown') for d in train_data])
        test_dist = Counter([d.get('ett_placement', 'Unknown') for d in test_data])
        print(f"  Train ({len(train_data)}): {dict(train_dist)}")
        print(f"  Test ({len(test_data)}): {dict(test_dist)}")

        # Create COCO format
        train_coco = create_coco_format(train_data, hospital_name, 'train')
        test_coco = create_coco_format(test_data, hospital_name, 'test')

        # Save individual hospital files
        train_file = os.path.join(OUTPUT_DIR, f"{hospital_name}-train-annotations.json")
        test_file = os.path.join(OUTPUT_DIR, f"{hospital_name}-test-annotations.json")

        with open(train_file, 'w') as f:
            json.dump(train_coco, f, indent=2)
        with open(test_file, 'w') as f:
            json.dump(test_coco, f, indent=2)

        # Aggregate for combined files
        all_train.extend(train_data)
        all_test.extend(test_data)

    # Create and save aggregate files
    print("\nCreating aggregate files...")
    all_train_coco = create_coco_format(all_train, 'All Hospitals', 'train')
    all_test_coco = create_coco_format(all_test, 'All Hospitals', 'test')

    with open(os.path.join(OUTPUT_DIR, "all-hospitals-train-annotations.json"), 'w') as f:
        json.dump(all_train_coco, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "all-hospitals-test-annotations.json"), 'w') as f:
        json.dump(all_test_coco, f, indent=2)

    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY:")
    print(f"  Total PROBLEM status filtered: {problem_count[0]}")
    print(f"  Total valid processed: {total_processed}")
    print(f"  Train images: {len(all_train)}")
    print(f"  Test images: {len(all_test)}")
    print(f"  Total hospitals: {len(hospital_data)}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
