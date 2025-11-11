#!/usr/bin/env python3
"""
Standardized ETT Width Annotation Utilities

This module provides a single source of truth for:
- Loading clinical annotations (including special hospital ID mapping)
- Extracting ETT width measurements
- Converting pixel distances to physical distances (cm/mm)
- Patient demographics lookup (weight, gestational age)

Usage:
    from ett_width_utils import (
        load_all_clinical_annotations,
        load_annotation_metadata,
        calculate_distance_with_ett_width_hierarchical,
        load_hospital_weights_standardized
    )
"""

import json
import re
import os
import unicodedata
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional


# ==============================================================================
# CONFIGURATION - Repository Data Paths
# ==============================================================================

# Get repository root directory (3 levels up from this file: analysis/utils/ett_width_utils.py)
REPO_ROOT = Path(__file__).parent.parent.parent

# Paths to data within the repository (relative to repo root)
DATA_DIR = REPO_ROOT / "data"
ANNOTATIONS_DIR = DATA_DIR / "annotations" / "preprocessed_640x640"
DEMOGRAPHICS_DIR = DATA_DIR / "demographics"
MAPPINGS_CSV = DATA_DIR / "mappings.csv"
CLINICAL_ANNOTATIONS_JSON = DATA_DIR / "clinical_annotations.json"

# For loading raw annotation sources (only needed if regenerating clinical_annotations.json)
# Can be overridden with ANNOTATION_SOURCE_DIR environment variable
ANNOTATION_SOURCE_DIR = os.environ.get('ANNOTATION_SOURCE_DIR', DATA_DIR / "annotation_source")


# ==============================================================================
# HOSPITAL NAME NORMALIZATION
# ==============================================================================

def normalize_hospital_name_to_csv_format(hospital_name: str) -> str:
    """
    Normalize hospital name from mappings.csv format to demographics CSV format.

    Handles edge cases where simple normalization doesn't work (typos, abbreviations, etc.)

    Args:
        hospital_name: Hospital name from mappings.csv (e.g., "Fundación Santa Fe de Bogotá")

    Returns:
        Normalized name matching CSV filenames (e.g., "Fundacion_Santa_Fe_de_Bogota")
    """
    # Strip trailing/leading whitespace
    hospital_name = hospital_name.strip()

    # First, check for special case mappings
    special_mappings = {
        # Hospitals with trailing spaces or missing suffixes
        'Indus': 'Indus_Hospital',
        'SES': 'SES_Hospital',
        'Istanbul Training Research': 'Istanbul_Training_Research_Hospital',
        'Kirikkale Hospital': 'Kirikkale_Hospital',  # Also handles trailing space

        # Abbreviated vs full names
        'Uni-Tubingen': 'University_of_Tubingen',

        # Completely different names
        'Medical Center of South': 'Medical_University_of_South_Carolina',

        # Typos in mappings.csv
        'Puerta del Mar University Hosptial': 'Puerta_del_Mar_University_Hospital',

        # Parentheses and special characters
        'National University (Singapore)': 'National_University_Hospital',

        # Missing suffixes
        'Newark Beth Israel': 'Newark_Beth_Israel_Medical_Center',
    }

    # Check if exact match in special mappings (before normalization)
    if hospital_name in special_mappings:
        return special_mappings[hospital_name]

    # Standard normalization for other hospitals
    # Remove accents
    hospital_name = ''.join(
        c for c in unicodedata.normalize('NFD', hospital_name)
        if unicodedata.category(c) != 'Mn'
    )

    # Remove apostrophes
    hospital_name = hospital_name.replace("'", "")

    # Replace spaces with underscores
    hospital_name = hospital_name.replace(" ", "_")

    return hospital_name


# ==============================================================================
# ANNOTATION LOADING
# ==============================================================================

def load_all_clinical_annotations() -> Tuple[Dict, Dict]:
    """
    Load ALL clinical annotations including special hospitals with ID mapping.

    Returns:
        Tuple of (clinical_annotations dict, special_mappings dict)
        - clinical_annotations: Maps image_id -> annotation dict
        - special_mappings: Maps (prefix, patient_num) -> annotation_name for special hospitals

    Special hospitals (Chiang Mai, Essen, Osaka) have different ID formats:
        - Test format: chiang_mai_p95-study-generated_id_XXXXX.png
        - Annotation format: Maharaj_Nakorn_Chiang_Mai_Hospital_p95

    Note: This function loads from raw annotation source files. For most use cases,
    you should load from the pre-computed clinical_annotations.json file instead.
    """
    clinical_annotations = {}

    # Base directory for latest annotations
    # Use ANNOTATION_SOURCE_DIR from configuration (can be overridden by environment variable)
    latest_dir = Path(ANNOTATION_SOURCE_DIR)

    # If directory doesn't exist, print warning
    if not latest_dir.exists():
        print(f"Warning: Annotation source directory not found at {latest_dir}")
        print(f"Note: For production use, clinical annotations are pre-computed in {CLINICAL_ANNOTATIONS_JSON}")
        print("This function is primarily used for regenerating clinical_annotations.json from source files.")
        # Use a placeholder that will cause graceful failure if needed
        latest_dir = Path("/path/to/annotation/source/files")

    # Load main annotation files
    main_files = [
        "tasks.json",           # Main file with most hospitals (2834 annotations)
        "alberta_merged.json"   # Alberta hospital (100 annotations)
    ]

    for filename in main_files:
        file_path = latest_dir / filename
        if file_path.exists():
            with open(file_path, 'r') as f:
                annotations = json.load(f)
                if isinstance(annotations, list):
                    for ann in annotations:
                        name = ann.get('name', '').replace('.png', '')
                        clinical_annotations[name] = ann

    # Load special hospital annotations with ID mapping
    # These use different naming conventions that need mapping
    special_hospitals = {
        'chiang_mai': {
            'file': latest_dir / 'chiang_mai.json',
            'test_prefix': 'chiang_mai_p',
            'ann_prefix': 'Maharaj_Nakorn_Chiang_Mai_Hospital_p'
        },
        'essen': {
            'file': latest_dir / 'essen.json',
            'test_prefix': 'essen_p',
            'ann_prefix': 'Universitätsklinikum_Essen_p'
        },
        'osaka': {
            'file': latest_dir / 'osaka.json',
            'test_prefix': 'osaka_p',
            'ann_prefix': 'Osaka_Metropolitan_University_p'
        }
    }

    special_mappings = {}
    for hosp_key, hosp_info in special_hospitals.items():
        if hosp_info['file'].exists():
            with open(hosp_info['file'], 'r') as f:
                annotations = json.load(f)
                for ann in annotations:
                    orig_name = ann.get('name', '').replace('.png', '')
                    # Store by original name
                    clinical_annotations[orig_name] = ann
                    # Extract patient number and create mapping
                    match = re.search(r'_p(\d+)$', orig_name)
                    if match:
                        patient_num = int(match.group(1))
                        special_mappings[(hosp_info['test_prefix'], patient_num)] = orig_name

    return clinical_annotations, special_mappings


def get_clinical_annotation(image_id: str, clinical_annotations: Dict, special_mappings: Dict) -> Optional[Dict]:
    """
    Get clinical annotation for an image, handling special hospital ID mappings.

    Args:
        image_id: Image identifier (without .png extension)
        clinical_annotations: Dict from load_all_clinical_annotations()
        special_mappings: Dict from load_all_clinical_annotations()

    Returns:
        Annotation dict or None if not found
    """
    # Try direct lookup first
    if image_id in clinical_annotations:
        return clinical_annotations[image_id]

    # Try special hospital mapping
    for prefix in ['chiang_mai_p', 'essen_p', 'osaka_p']:
        if image_id.startswith(prefix):
            match = re.match(rf'{prefix}(\d+)-', image_id)
            if match:
                patient_num = int(match.group(1))
                mapped_name = special_mappings.get((prefix, patient_num))
                if mapped_name:
                    return clinical_annotations.get(mapped_name)

    return None


def extract_ett_width_pixels(annotation: Dict) -> Optional[float]:
    """
    Extract ETT width measurement in pixels from annotation (original image space).

    Args:
        annotation: Clinical annotation dict

    Returns:
        ETT width in pixels (original image space) or None if not found
    """
    if 'series' not in annotation:
        return None

    # Look for measurement with category='ETT width'
    for series in annotation['series']:
        measurements = series.get('measurements', [])
        for meas in measurements:
            if meas.get('category') == 'ETT width':
                # The value is stored in 'length' field
                if 'length' in meas:
                    return float(meas['length'])

    return None


def load_annotation_metadata(annotation_path: Path) -> Dict:
    """
    Load image metadata from COCO annotation file (orig dimensions, scale factors, etc.).

    Args:
        annotation_path: Path to COCO annotation JSON file

    Returns:
        Dict mapping file_name -> metadata dict with keys:
            - hospital_name
            - orig_width, orig_height (original dimensions)
            - width, height (resized dimensions, typically 640x640)
    """
    with open(annotation_path, 'r') as f:
        data = json.load(f)

    metadata_map = {}
    for img in data['images']:
        metadata_map[img['file_name']] = {
            'hospital_name': img.get('hospital_name'),
            'orig_width': img.get('orig_width', 640),
            'orig_height': img.get('orig_height', 640),
            'width': img.get('width', 640),
            'height': img.get('height', 640)
        }

    return metadata_map


# ==============================================================================
# PATIENT DEMOGRAPHICS LOOKUP
# ==============================================================================

def load_hospital_weights_standardized(csv_dir: Path) -> Dict:
    """
    Load standardized demographics CSVs with patient weights and gestational ages.

    Args:
        csv_dir: Directory containing *_nicu_settings.csv files

    Returns:
        Dict mapping hospital_name -> DataFrame with columns from CSV:
            - Folder (patient ID)
            - Weight (in kg)
            - Gestational Age (in weeks, may have format like "37 w 3D")
    """
    csv_dir = Path(csv_dir)
    hospital_weights = {}

    for csv_file in csv_dir.glob("*_nicu_settings.csv"):
        hospital_name = csv_file.stem.replace('_nicu_settings', '')
        try:
            df = pd.read_csv(csv_file)
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()
            hospital_weights[hospital_name] = df
        except Exception as e:
            print(f"Warning: Could not load {csv_file.name}: {e}")
            continue

    return hospital_weights


def get_patient_weight_standardized(random_id: str, mappings: pd.DataFrame,
                                   hospital_weights: Dict) -> Optional[float]:
    """
    Get patient weight in grams using standardized demographics files.

    Handles special hospital naming conventions for Alberta, Chiang Mai, Essen, Osaka.

    Args:
        random_id: Random image ID (e.g., 'random-id-XXX' or 'alberta-p57-study-XXX')
        mappings: DataFrame with hospital_name and random_id columns
        hospital_weights: Dict from load_hospital_weights_standardized()

    Returns:
        Patient weight in grams, or None if not found
    """
    # Map CSV file names to hospital names in mappings
    hospital_csv_mapping = {
        'Alberta': 'University_of_Alberta',
        'Maharaj-Nakorn-Chiang-Mai-Hospital': 'Maharaj_Nakorn_Chiang_Mai_Hospital',
        'Osaka-Metropolitan-University': 'Osaka_Metropolitan_University_Hospital',
        'Universitaetsklinikum-Essen': 'Essen_University_Hospital'
    }

    # Handle special hospital prefixes
    patient_num = None
    hospital_csv_name = None

    if random_id.startswith('alberta-p') or random_id.startswith('alberta_p'):
        match = re.match(r'alberta[-_]p(\d+)-', random_id)
        if match:
            patient_num = int(match.group(1))
            hospital_csv_name = 'University_of_Alberta'

    elif random_id.startswith('chiang_mai_p') or random_id.startswith('chiang_p'):
        match = re.match(r'chiang(?:_mai)?_p(\d+)-', random_id)
        if match:
            patient_num = int(match.group(1))
            hospital_csv_name = 'Maharaj_Nakorn_Chiang_Mai_Hospital'

    elif random_id.startswith('essen_p'):
        match = re.match(r'essen_p(\d+)-', random_id)
        if match:
            patient_num = int(match.group(1))
            hospital_csv_name = 'Essen_University_Hospital'

    elif random_id.startswith('osaka_p'):
        match = re.match(r'osaka_p(\d+)-', random_id)
        if match:
            patient_num = int(match.group(1))
            hospital_csv_name = 'Osaka_Metropolitan_University_Hospital'

    else:
        # Standard random ID - look up in mappings
        matched_rows = mappings[mappings['new_name'] == random_id]
        if len(matched_rows) > 0:
            hospital_name = matched_rows.iloc[0]['institution']
            original_name = matched_rows.iloc[0]['original_name']

            # Normalize hospital name to match CSV format
            hospital_csv_name = normalize_hospital_name_to_csv_format(hospital_name)

            # Extract patient number from original name (e.g., p54-study... → 54)
            patient_match = re.match(r'p(\d+)-', original_name)
            if patient_match:
                patient_num = int(patient_match.group(1))

    # Look up weight in hospital demographics
    if hospital_csv_name and hospital_csv_name in hospital_weights:
        df = hospital_weights[hospital_csv_name]

        if patient_num is not None and 'Folder' in df.columns:
            # Match by Folder (patient number)
            # Folder can be just a number or "p74" format

            # Check for weight column (could be 'Weight (grams)' or 'Weight (gram)')
            weight_col = None
            if 'Weight (grams)' in df.columns:
                weight_col = 'Weight (grams)'
            elif 'Weight (gram)' in df.columns:
                weight_col = 'Weight (gram)'

            if weight_col:
                for _, row in df.iterrows():
                    if pd.notna(row['Folder']):
                        folder_str = str(row['Folder']).strip()
                        # Handle both 'p74' and '74' formats
                        folder_num = folder_str.replace('p', '') if folder_str.startswith('p') else folder_str
                        try:
                            folder_int = int(float(folder_num))
                            if folder_int == patient_num:
                                # Found matching patient, get weight
                                if pd.notna(row[weight_col]):
                                    weight_val = float(row[weight_col])
                                    # Weight is in grams in these CSVs
                                    return weight_val
                        except (ValueError, TypeError):
                            continue

    return None


def get_patient_gestational_age(random_id: str, mappings: pd.DataFrame,
                                hospital_weights: Dict) -> Optional[float]:
    """
    Get patient gestational age in weeks using standardized demographics files.

    Args:
        random_id: Random image ID
        mappings: DataFrame with hospital_name and random_id columns
        hospital_weights: Dict from load_hospital_weights_standardized()

    Returns:
        Gestational age in weeks, or None if not found
    """
    # Similar logic to get_patient_weight_standardized
    hospital_csv_mapping = {
        'Alberta': 'University_of_Alberta',
        'Maharaj-Nakorn-Chiang-Mai-Hospital': 'Maharaj_Nakorn_Chiang_Mai_Hospital',
        'Osaka-Metropolitan-University': 'Osaka_Metropolitan_University_Hospital',
        'Universitaetsklinikum-Essen': 'Essen_University_Hospital'
    }

    patient_num = None
    hospital_csv_name = None

    if random_id.startswith('alberta-p') or random_id.startswith('alberta_p'):
        match = re.match(r'alberta[-_]p(\d+)-', random_id)
        if match:
            patient_num = int(match.group(1))
            hospital_csv_name = 'University_of_Alberta'

    elif random_id.startswith('chiang_mai_p') or random_id.startswith('chiang_p'):
        match = re.match(r'chiang(?:_mai)?_p(\d+)-', random_id)
        if match:
            patient_num = int(match.group(1))
            hospital_csv_name = 'Maharaj_Nakorn_Chiang_Mai_Hospital'

    elif random_id.startswith('essen_p'):
        match = re.match(r'essen_p(\d+)-', random_id)
        if match:
            patient_num = int(match.group(1))
            hospital_csv_name = 'Essen_University_Hospital'

    elif random_id.startswith('osaka_p'):
        match = re.match(r'osaka_p(\d+)-', random_id)
        if match:
            patient_num = int(match.group(1))
            hospital_csv_name = 'Osaka_Metropolitan_University_Hospital'

    else:
        matched_rows = mappings[mappings['new_name'] == random_id]
        if len(matched_rows) > 0:
            hospital_name = matched_rows.iloc[0]['institution']
            original_name = matched_rows.iloc[0]['original_name']

            # Normalize hospital name to match CSV format
            hospital_csv_name = normalize_hospital_name_to_csv_format(hospital_name)

            # Extract patient number from original name (e.g., p54-study... → 54)
            patient_match = re.match(r'p(\d+)-', original_name)
            if patient_match:
                patient_num = int(patient_match.group(1))

    # Look up GA in hospital demographics
    if hospital_csv_name and hospital_csv_name in hospital_weights:
        df = hospital_weights[hospital_csv_name]

        if patient_num is not None and 'Folder' in df.columns:
            # Match by Folder (patient number)
            for _, row in df.iterrows():
                if pd.notna(row['Folder']):
                    folder_str = str(row['Folder']).strip()
                    # Handle both 'p74' and '74' formats
                    folder_num = folder_str.replace('p', '') if folder_str.startswith('p') else folder_str
                    try:
                        folder_int = int(float(folder_num))
                        if folder_int == patient_num:
                            # Found matching patient, get gestational age
                            if 'Gestational Age' in row and pd.notna(row['Gestational Age']):
                                ga_str = str(row['Gestational Age']).strip()
                                # Parse format like "37 w 3D" or "37W" or just "37"
                                match = re.match(r'(\d+)\s*[wW]?', ga_str)
                                if match:
                                    ga_weeks = float(match.group(1))
                                    return ga_weeks
                    except (ValueError, TypeError):
                        continue

    return None


# ==============================================================================
# ETT WIDTH HIERARCHICAL FALLBACK
# ==============================================================================

def get_ett_width_from_weight(weight_grams: float) -> float:
    """
    Determine ETT width (mm) based on patient weight.

    Clinical guidelines for ETT sizing based on weight.

    Args:
        weight_grams: Patient weight in grams

    Returns:
        ETT width in millimeters
    """
    if weight_grams < 1000:
        return 2.5   # < 1000g (<1 kg)
    elif weight_grams < 2000:
        return 3.0   # 1000-2000g (1-2 kg)
    elif weight_grams < 3000:
        return 3.5   # 2000-3000g (2-3 kg)
    else:
        return 3.75  # >= 3000g (>= 3 kg)


def get_ett_width_from_gestational_age(ga_weeks: float) -> float:
    """
    Determine ETT width (mm) based on gestational age (fallback method).

    Args:
        ga_weeks: Gestational age in weeks

    Returns:
        ETT width in millimeters
    """
    if ga_weeks < 28:
        return 2.5   # Extremely preterm
    elif ga_weeks < 32:
        return 3.0   # Very preterm
    elif ga_weeks < 37:
        return 3.5   # Late preterm
    else:
        return 3.75  # Term


# ==============================================================================
# DISTANCE CALCULATION
# ==============================================================================

def calculate_pixel_distance(point1: Tuple, point2: Tuple) -> Optional[float]:
    """
    Calculate Euclidean distance between two points in pixel space.

    Args:
        point1: (x, y) coordinates
        point2: (x, y) coordinates

    Returns:
        Distance in pixels, or None if invalid points
    """
    if point1 is None or point2 is None:
        return None
    if len(point1) < 2 or len(point2) < 2:
        return None

    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def convert_pixels_to_cm(pixel_distance: float, ett_width_pixels: float,
                         ett_width_mm: float) -> Optional[float]:
    """
    Convert pixel distance to centimeters using ETT width as reference.

    Args:
        pixel_distance: Distance in pixels (640x640 space)
        ett_width_pixels: ETT width in pixels (640x640 space)
        ett_width_mm: ETT width in millimeters (physical size)

    Returns:
        Distance in centimeters, or None if invalid
    """
    if ett_width_pixels <= 0 or ett_width_mm <= 0:
        return None

    # mm per pixel = ETT width in mm / ETT width in pixels
    mm_per_pixel = ett_width_mm / ett_width_pixels

    # Convert pixel distance to mm, then to cm
    distance_mm = pixel_distance * mm_per_pixel
    distance_cm = distance_mm / 10.0

    return distance_cm


def calculate_distance_with_ett_width_hierarchical(
    point1: Tuple,
    point2: Tuple,
    image_name: str,
    metadata: Dict,
    clinical_annotations: Dict,
    special_mappings: Dict,
    mappings: pd.DataFrame,
    hospital_weights: Dict
) -> float:
    """
    Calculate physical distance using hierarchical ETT width conversion.

    Hierarchy:
        1. Weight-based ETT width (most accurate)
        2. Gestational age-based ETT width (fallback)
        3. Default 3.5mm ETT width (final fallback)

    Args:
        point1, point2: Coordinates in 640x640 space
        image_name: Image filename
        metadata: Image metadata dict (orig_width, orig_height, etc.)
        clinical_annotations: Dict from load_all_clinical_annotations()
        special_mappings: Dict from load_all_clinical_annotations()
        mappings: Hospital mappings DataFrame
        hospital_weights: Demographics dict

    Returns:
        Distance in millimeters (or np.nan if cannot calculate)
    """
    if point1 is None or point2 is None:
        return np.nan

    # Calculate pixel distance in 640x640 space
    pixel_distance = calculate_pixel_distance(point1, point2)
    if pixel_distance is None:
        return np.nan

    # Get image ID and clinical annotation
    image_id = image_name.replace('.png', '')
    clinical_ann = get_clinical_annotation(image_id, clinical_annotations, special_mappings)

    if clinical_ann:
        # Get ETT width in original image space
        ett_width_pixels_orig = extract_ett_width_pixels(clinical_ann)

        if ett_width_pixels_orig:
            # Hierarchical ETT width determination
            ett_width_mm = None

            # Try weight first (primary method)
            patient_weight = get_patient_weight_standardized(image_id, mappings, hospital_weights)
            if patient_weight is not None:
                ett_width_mm = get_ett_width_from_weight(patient_weight)
            else:
                # Try gestational age (fallback)
                patient_ga = get_patient_gestational_age(image_id, mappings, hospital_weights)
                if patient_ga is not None:
                    ett_width_mm = get_ett_width_from_gestational_age(patient_ga)
                else:
                    # Default fallback
                    ett_width_mm = 3.5

            # Calculate scale factor from original to 640x640
            orig_width = metadata.get('orig_width', 640)
            orig_height = metadata.get('orig_height', 640)
            scale_factor = 640 / max(orig_width, orig_height)

            # Scale ETT width to 640x640 space
            ett_width_pixels_640 = ett_width_pixels_orig * scale_factor

            # Convert pixel distance to cm using ETT width
            distance_cm = convert_pixels_to_cm(pixel_distance, ett_width_pixels_640, ett_width_mm)

            if distance_cm is not None:
                return distance_cm * 10  # Convert cm to mm for consistency

    # Final fallback: use default 3.5mm ETT width
    # This ensures we ALWAYS use ETT width calibration, never pixel spacing
    # Assume a reasonable ETT width in pixels if annotation missing (40 pixels is typical)
    default_ett_width_pixels = 40.0
    default_ett_width_mm = 3.5
    distance_cm = convert_pixels_to_cm(pixel_distance, default_ett_width_pixels, default_ett_width_mm)

    if distance_cm is not None:
        return distance_cm * 10  # Convert to mm

    return np.nan


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def parse_coordinates(coord_str: str) -> Optional[Tuple[float, float]]:
    """
    Parse coordinate string to (x, y) tuple.

    Handles formats: "(x, y)" or "x,y" or "[x, y]"

    Args:
        coord_str: String representation of coordinates

    Returns:
        (x, y) tuple or None if invalid
    """
    if pd.isna(coord_str) or coord_str == '' or coord_str == 'nan':
        return None

    try:
        # Remove brackets/parentheses and split
        coord_str = str(coord_str).strip('()[]')
        parts = [float(x.strip()) for x in coord_str.split(',')]
        if len(parts) >= 2:
            return (parts[0], parts[1])
    except (ValueError, AttributeError):
        return None

    return None


def normalize_hospital_name(hospital_name: str) -> str:
    """
    Normalize hospital names to ensure consistency.

    This matches the normalization in compare_cl_vs_finetuning_horizontal_ett_width.py

    Args:
        hospital_name: Raw hospital name from data

    Returns:
        Normalized hospital name with hyphens
    """
    if pd.isna(hospital_name) or hospital_name == 'Unknown':
        return hospital_name

    hospital_name = str(hospital_name).strip()

    # Special handling for Kirikkale (remove trailing '- ')
    if 'Kirikkale' in hospital_name:
        hospital_name = hospital_name.rstrip('- ')
        return 'Kirikkale-Hospital'

    # Replace spaces with hyphens (key difference!)
    hospital_name = hospital_name.replace(' ', '-')

    # Special cases
    if 'Children' in hospital_name and 'Colorado' in hospital_name:
        return 'Childrens-Hospital-Colorado'
    if 'Fundacion' in hospital_name or 'Bogota' in hospital_name:
        return 'Fundacion-Santa-Fe-de-Bogota'

    return hospital_name
