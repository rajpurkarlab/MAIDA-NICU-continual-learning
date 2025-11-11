#!/usr/bin/env python3
"""
Standardized Prediction Loading Utilities

This module provides a single source of truth for loading predictions from:
- Continual Learning (CL) holdout results (3 locations, 10 simulations each)
- Fine-tuning (FT) results (10 simulations)
- Direct Prediction (inference) results (single simulation)

Two levels of functions:
- Level 1 (Raw): Load CSV files into structured dictionaries
- Level 2 (Processed): Load + calculate distances + aggregate statistics

Usage:
    from prediction_loading_utils import (
        load_cl_predictions_multi_sim,
        load_ft_predictions_multi_sim,
        load_direct_predictions,
        process_cl_results_multi_sim,
        process_ft_results_multi_sim,
        process_direct_results
    )
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
from scipy import stats as scipy_stats

# Import from ett_width_utils
from ett_width_utils import (
    parse_coordinates,
    normalize_hospital_name,
    calculate_distance_with_ett_width_hierarchical
)


# ==============================================================================
# CONFIGURATION - MODIFY THESE PATHS FOR YOUR SETUP
# ==============================================================================

# IMPORTANT: Update these paths to match your experiment output locations
# These paths should point to where your continual learning, fine-tuning,
# and direct prediction experiments saved their results.

# Continual Learning Output Locations
# If you ran CL experiments in multiple locations (recommended for parallel execution),
# specify each location and which hospitals were processed there.
# If you ran everything in one location, use only one entry.
CL_LOCATIONS = {
    'LOCATION_1': {
        'path': '/path/to/your/continual_learning_outputs/location_1',  # UPDATE THIS PATH
        'hospitals': [
            'Alberta', 'American-University-of-Beirut', 'Childrens-Hospital-Colorado',
            'Chulalongkorn-University', 'Dr-Sardjito-Hospital', 'Fundacion-Santa-Fe-de-Bogota',
            'Indus', 'International-Islamic-Medical-University', 'Istanbul-Training-Research',
            'King-Abdulaziz-Hospital', 'Kirikkale-Hospital-'
        ]
    },
    'LOCATION_2': {
        'path': '/path/to/your/continual_learning_outputs/location_2',  # UPDATE THIS PATH
        'hospitals': [
            'Sichuan-People-Hospital', 'Sidra-Health', 'Tel-Aviv-Medical-Center',
            'Tri-Service-General-Hospital', 'Uni-Tubingen', 'Universitaetsklinikum-Essen',
            'University-Hospital-Aachen', 'University-of-Graz', 'University-of-Kragujevac',
            'University-of-Linz'
        ]
    },
    'LOCATION_3': {
        'path': '/path/to/your/continual_learning_outputs/location_3',  # UPDATE THIS PATH
        'hospitals': [
            'La-Paz-University-Hospital', 'Maharaj-Nakorn-Chiang-Mai-Hospital',
            'Medical-Center-of-South', 'National-Cheng-Kung-University-Hospital',
            'National-University-Singapore', 'Newark-Beth-Israel',
            'Osaka-Metropolitan-University', 'Puerta-del-Mar-University-Hosptial',
            'SES', 'Shiraz-University'
        ]
    }
}

# Fine-Tuning Output Directory
# Path to directory containing fine-tuning results organized by simulation
# Expected structure: {DEFAULT_FT_DIR}/simulation_0/finetuned-{hospital}.csv
DEFAULT_FT_DIR = Path("/path/to/your/fine_tuning_outputs")  # UPDATE THIS PATH

# Direct Prediction Results Path
# Path to the direct prediction CSV (pretrained model without adaptation)
# Expected structure: {DEFAULT_DIRECT_PATH} containing predictions for all hospitals
DEFAULT_DIRECT_PATH = Path("/path/to/your/direct_prediction/nicu_test_predictions.csv")  # UPDATE THIS PATH


# ==============================================================================
# LEVEL 1: RAW PREDICTION LOADING (Returns Dicts)
# ==============================================================================

def load_single_prediction_csv(csv_path: Path) -> Dict:
    """
    Load a single prediction CSV file.

    Args:
        csv_path: Path to CSV file with columns: file_name, category, pred, [gPoint]

    Returns:
        Dict mapping image_id -> {
            'file_name': str,
            'hospital_name': str,
            'tip_pred': (x, y) or None,
            'carina_pred': (x, y) or None,
            'tip_gt': (x, y) or None (if available),
            'carina_gt': (x, y) or None (if available)
        }
    """
    if not csv_path.exists():
        return {}

    df = pd.read_csv(csv_path)

    # Parse coordinates
    df['pred_coords'] = df['pred'].apply(parse_coordinates)
    if 'gPoint' in df.columns:
        df['gt_coords'] = df['gPoint'].apply(parse_coordinates)

    predictions = {}

    for file_name, group in df.groupby('file_name'):
        # Get hospital name from CSV
        if 'hospital_name' in group.columns:
            hospital_name = group['hospital_name'].iloc[0]
        elif 'index_hospital' in group.columns:
            hospital_name = group['index_hospital'].iloc[0]
        else:
            hospital_name = 'Unknown'

        # Normalize hospital name (remove parentheses, standardize format)
        hospital_name = re.sub(r'[()]', '', hospital_name)
        hospital_name = normalize_hospital_name(hospital_name)

        # Get tip and carina predictions
        tip_data = group[group['category'] == 'tip']
        carina_data = group[group['category'] == 'carina']

        prediction = {
            'file_name': file_name,
            'hospital_name': hospital_name,
            'tip_pred': None,
            'carina_pred': None,
            'tip_gt': None,
            'carina_gt': None
        }

        if len(tip_data) > 0:
            tip_row = tip_data.iloc[0]
            prediction['tip_pred'] = tip_row['pred_coords']
            if 'gt_coords' in tip_row:
                prediction['tip_gt'] = tip_row['gt_coords']

        if len(carina_data) > 0:
            carina_row = carina_data.iloc[0]
            prediction['carina_pred'] = carina_row['pred_coords']
            if 'gt_coords' in carina_row:
                prediction['carina_gt'] = carina_row['gt_coords']

        image_id = file_name.replace('.png', '')
        predictions[image_id] = prediction

    return predictions


def load_cl_predictions_multi_sim(
    locations: Optional[Dict] = None,
    num_simulations: int = 10
) -> Dict[str, List[Dict]]:
    """
    Load CL holdout predictions from all simulations across 3 locations.

    Args:
        locations: Dict with location info (defaults to CL_LOCATIONS)
        num_simulations: Number of simulations to load (default: 10)

    Returns:
        Dict mapping hospital_name -> list of prediction dicts (one per simulation)
        Each prediction dict maps image_id -> prediction info
    """
    if locations is None:
        locations = CL_LOCATIONS

    hospital_simulations = {}
    missing_simulations = []

    for loc_name, loc_info in locations.items():
        base_path = Path(loc_info['path'])

        if not base_path.exists():
            print(f"     {loc_name} path not found: {base_path}")
            continue

        print(f"    Processing {loc_name} location...")

        for hospital in loc_info['hospitals']:
            hospital_normalized = normalize_hospital_name(hospital)

            # Initialize list for this hospital
            if hospital_normalized not in hospital_simulations:
                hospital_simulations[hospital_normalized] = []

            # Check for holdout folder
            holdout_folder = base_path / f"{hospital}_holdout"

            if not holdout_folder.exists():
                print(f"        {hospital}: Holdout folder not found")
                continue

            # Process all simulations
            for sim_num in range(num_simulations):
                sim_dir = holdout_folder / f"simulation_{sim_num}"
                csv_file = sim_dir / "holdout_test_results.csv"

                if csv_file.exists():
                    predictions = load_single_prediction_csv(csv_file)
                    hospital_simulations[hospital_normalized].append(predictions)
                else:
                    missing_simulations.append(f"{loc_name}/{hospital}/simulation_{sim_num}")
                    print(f"        Missing: {hospital} simulation_{sim_num}")

    if missing_simulations:
        print(f"\n     Total missing simulations: {len(missing_simulations)}")

    # Report summary
    print(f"\n    Loaded CL predictions for {len(hospital_simulations)} hospitals")
    for hospital, sims in hospital_simulations.items():
        if len(sims) != num_simulations:
            print(f"        {hospital}: {len(sims)}/{num_simulations} simulations")

    return hospital_simulations


def load_ft_predictions_multi_sim(
    ft_dir: Optional[Path] = None,
    num_simulations: int = 10
) -> Dict[str, List[Dict]]:
    """
    Load fine-tuning predictions from all simulations.

    Args:
        ft_dir: Directory containing simulation_0/ ... simulation_9/ folders
        num_simulations: Number of simulations to load (default: 10)

    Returns:
        Dict mapping hospital_name -> list of prediction dicts (one per simulation)
    """
    if ft_dir is None:
        ft_dir = DEFAULT_FT_DIR

    ft_dir = Path(ft_dir)
    hospital_simulations = {}
    missing_simulations = []

    if not ft_dir.exists():
        print(f"    Fine-tuning directory not found: {ft_dir}")
        return hospital_simulations

    print(f"    Processing fine-tuning results from {num_simulations} simulations...")

    # Process all simulations
    for sim_num in range(num_simulations):
        sim_dir = ft_dir / f"simulation_{sim_num}"

        if not sim_dir.exists():
            print(f"        simulation_{sim_num} directory not found")
            missing_simulations.append(f"simulation_{sim_num}")
            continue

        # Process each hospital CSV file in this simulation
        for csv_file in sim_dir.glob("finetuned-*.csv"):
            hospital_name = csv_file.stem.replace('finetuned-', '')
            hospital_normalized = normalize_hospital_name(hospital_name)

            # Initialize list for this hospital
            if hospital_normalized not in hospital_simulations:
                hospital_simulations[hospital_normalized] = []

            # Load predictions
            predictions = load_single_prediction_csv(csv_file)
            hospital_simulations[hospital_normalized].append(predictions)

    if missing_simulations:
        print(f"\n     Missing simulation directories: {', '.join(missing_simulations)}")

    # Report summary
    print(f"\n    Loaded FT predictions for {len(hospital_simulations)} hospitals")
    for hospital, sims in hospital_simulations.items():
        if len(sims) != num_simulations:
            print(f"        {hospital}: {len(sims)}/{num_simulations} simulations")

    return hospital_simulations


def load_direct_predictions(
    direct_path: Optional[Path] = None,
    mappings: Optional[pd.DataFrame] = None
) -> Dict[str, Dict]:
    """
    Load direct prediction results (inference without fine-tuning).

    Direct predictions have all hospitals in one CSV marked as "All Hospitals".
    This function maps them to actual hospital names.

    Args:
        direct_path: Path to direct predictions CSV
        mappings: DataFrame with hospital mappings (optional, for mapping lookup)

    Returns:
        Dict mapping image_id -> prediction info (with correct hospital_name)
    """
    if direct_path is None:
        direct_path = DEFAULT_DIRECT_PATH

    direct_path = Path(direct_path)

    if not direct_path.exists():
        print(f"    Direct predictions file not found: {direct_path}")
        return {}

    print(f"    Loading direct predictions from: {direct_path.name}")

    df = pd.read_csv(direct_path)

    # Map "All Hospitals" to actual hospital names
    def get_actual_hospital(file_name: str) -> str:
        """Map file name to actual hospital name.

        This matches the logic in FINAL_create_etable5_multi_sim_with_alberta_broken.py
        """
        file_id = file_name.replace('.png', '')

        # Check special hospital prefixes (return hyphenated names to match normalize)
        if file_id.startswith('essen_'):
            return 'Universitaetsklinikum-Essen'
        elif file_id.startswith('osaka_'):
            return 'Osaka-Metropolitan-University'
        elif file_id.startswith('chiang'):  # Matches 'chiang_mai_p' files
            return 'Maharaj-Nakorn-Chiang-Mai-Hospital'
        elif file_id.startswith('alberta-p') or file_id.startswith('alberta_p'):  # FIXED: both hyphen and underscore
            return 'Alberta'

        # For all other files, look up in mappings
        # Try mappings as dict first
        if mappings is not None and isinstance(mappings, dict) and file_id in mappings:
            institution = mappings[file_id].get('institution', 'Unknown')
            institution_clean = re.sub(r'[()]', '', institution)
            normalized = normalize_hospital_name(institution_clean)
            return normalized

        # Try mappings as DataFrame - FIXED: check 'new_name' column, not index
        if mappings is not None and hasattr(mappings, 'columns') and 'new_name' in mappings.columns:
            matched_rows = mappings[mappings['new_name'] == file_id]
            if len(matched_rows) > 0:
                institution = matched_rows.iloc[0]['institution']
                institution_clean = re.sub(r'[()]', '', institution)
                normalized = normalize_hospital_name(institution_clean)
                return normalized

        return 'Unknown'

    # Apply hospital mapping if needed
    if 'hospital_name' in df.columns and (df['hospital_name'] == 'All Hospitals').any():
        print(f"      → Mapping 'All Hospitals' to actual hospital names...")
        mask = df['hospital_name'] == 'All Hospitals'
        df.loc[mask, 'hospital_name'] = df.loc[mask, 'file_name'].apply(get_actual_hospital)
        mapped_count = mask.sum()
        print(f"       Mapped {mapped_count} predictions to actual hospitals")

    # Parse coordinates
    df['pred_coords'] = df['pred'].apply(parse_coordinates)
    if 'gPoint' in df.columns:
        df['gt_coords'] = df['gPoint'].apply(parse_coordinates)

    predictions = {}
    hospitals_found = set()

    for file_name, group in df.groupby('file_name'):
        hospital_name = group['hospital_name'].iloc[0]
        hospital_name = re.sub(r'[()]', '', hospital_name)
        hospital_name = normalize_hospital_name(hospital_name)
        hospitals_found.add(hospital_name)

        tip_data = group[group['category'] == 'tip']
        carina_data = group[group['category'] == 'carina']

        prediction = {
            'file_name': file_name,
            'hospital_name': hospital_name,
            'tip_pred': None,
            'carina_pred': None,
            'tip_gt': None,
            'carina_gt': None
        }

        if len(tip_data) > 0:
            tip_row = tip_data.iloc[0]
            prediction['tip_pred'] = tip_row['pred_coords']
            if 'gt_coords' in tip_row:
                prediction['tip_gt'] = tip_row['gt_coords']

        if len(carina_data) > 0:
            carina_row = carina_data.iloc[0]
            prediction['carina_pred'] = carina_row['pred_coords']
            if 'gt_coords' in carina_row:
                prediction['carina_gt'] = carina_row['gt_coords']

        image_id = file_name.replace('.png', '')
        predictions[image_id] = prediction

    print(f"    Loaded {len(predictions)} direct predictions across {len(hospitals_found)} hospitals")

    return predictions


# ==============================================================================
# LEVEL 2: PROCESSED RESULTS (Returns DataFrames with Distances)
# ==============================================================================

def calculate_errors_for_predictions(
    predictions: Dict,
    metadata_map: Dict,
    clinical_annotations: Dict,
    special_mappings: Dict,
    mappings: pd.DataFrame,
    hospital_weights: Dict
) -> pd.DataFrame:
    """
    Calculate tip, carina, and combined errors for a set of predictions.

    Args:
        predictions: Dict from load_single_prediction_csv()
        metadata_map: Image metadata dict
        clinical_annotations: Clinical annotations dict
        special_mappings: Special hospital mappings dict
        mappings: Hospital mappings DataFrame
        hospital_weights: Patient demographics dict

    Returns:
        DataFrame with columns:
            - file_name
            - hospital_name
            - tip_error_mm
            - carina_error_mm
            - combined_tip_carina_error_mm
    """
    results = []

    for image_id, pred in predictions.items():
        file_name = pred['file_name']
        hospital_name = pred['hospital_name']

        # Get metadata
        metadata = metadata_map.get(file_name, {
            'orig_width': 640, 'orig_height': 640,
            'width': 640, 'height': 640
        })

        result = {
            'file_name': file_name,
            'hospital_name': hospital_name,
            'tip_error_mm': np.nan,
            'carina_error_mm': np.nan,
            'combined_tip_carina_error_mm': np.nan
        }

        # Calculate tip error
        if pred['tip_pred'] is not None and pred['tip_gt'] is not None:
            tip_error = calculate_distance_with_ett_width_hierarchical(
                pred['tip_pred'], pred['tip_gt'], file_name, metadata,
                clinical_annotations, special_mappings, mappings, hospital_weights
            )
            result['tip_error_mm'] = tip_error

        # Calculate carina error
        if pred['carina_pred'] is not None and pred['carina_gt'] is not None:
            carina_error = calculate_distance_with_ett_width_hierarchical(
                pred['carina_pred'], pred['carina_gt'], file_name, metadata,
                clinical_annotations, special_mappings, mappings, hospital_weights
            )
            result['carina_error_mm'] = carina_error

        # Calculate combined error
        if not np.isnan(result['tip_error_mm']) and not np.isnan(result['carina_error_mm']):
            result['combined_tip_carina_error_mm'] = result['tip_error_mm'] + result['carina_error_mm']

        results.append(result)

    return pd.DataFrame(results)


def aggregate_simulation_stats(
    hospital_simulation_errors: Dict[str, List[float]],
    use_median: bool = True
) -> pd.DataFrame:
    """
    Aggregate errors across simulations with mean, std, and 95% CI.

    Args:
        hospital_simulation_errors: Dict[hospital -> list of errors per simulation]
        use_median: If True, also calculate median (default: True)

    Returns:
        DataFrame with columns:
            - hospital_name
            - n_simulations
            - combined_error_mean_mm
            - combined_error_std_mm
            - combined_error_ci_margin_mm (95% CI margin)
            - combined_error_median_mm (optional)
    """
    results = []

    for hospital, errors in hospital_simulation_errors.items():
        if not errors:
            continue

        errors_array = np.array(errors)
        n = len(errors_array)
        mean_error = np.mean(errors_array)
        std_error = np.std(errors_array, ddof=1) if n > 1 else 0.0

        # Calculate 95% confidence interval
        if n > 1:
            sem = std_error / np.sqrt(n)
            t_critical = scipy_stats.t.ppf(0.975, n - 1)
            ci_margin = t_critical * sem
        else:
            ci_margin = 0.0

        result = {
            'hospital_name': hospital,
            'n_simulations': n,
            'combined_error_mean_mm': mean_error,
            'combined_error_std_mm': std_error,
            'combined_error_ci_margin_mm': ci_margin
        }

        if use_median:
            result['combined_error_median_mm'] = np.median(errors_array)

        results.append(result)

    return pd.DataFrame(results)


def process_cl_results_multi_sim(
    metadata_map: Dict,
    clinical_annotations: Dict,
    special_mappings: Dict,
    mappings: pd.DataFrame,
    hospital_weights: Dict,
    locations: Optional[Dict] = None,
    num_simulations: int = 10
) -> pd.DataFrame:
    """
    Load CL predictions, calculate distances, and aggregate statistics.

    Args:
        metadata_map: Image metadata from load_annotation_metadata()
        clinical_annotations: From load_all_clinical_annotations()
        special_mappings: From load_all_clinical_annotations()
        mappings: Hospital mappings DataFrame
        hospital_weights: Patient demographics dict
        locations: CL locations dict (defaults to CL_LOCATIONS)
        num_simulations: Number of simulations (default: 10)

    Returns:
        DataFrame with aggregated statistics per hospital:
            - hospital_name
            - n_simulations
            - combined_error_mean_mm
            - combined_error_std_mm
            - combined_error_ci_margin_mm
            - combined_error_median_mm
    """
    print("\n Processing CL holdout results...")

    # Load raw predictions
    hospital_simulations = load_cl_predictions_multi_sim(locations, num_simulations)

    # Calculate errors for each hospital/simulation
    hospital_simulation_errors = {}

    for hospital, sim_predictions_list in hospital_simulations.items():
        hospital_simulation_errors[hospital] = []

        for sim_idx, predictions in enumerate(sim_predictions_list):
            # Calculate errors for this simulation
            error_df = calculate_errors_for_predictions(
                predictions, metadata_map, clinical_annotations,
                special_mappings, mappings, hospital_weights
            )

            # Calculate median combined error for this simulation
            if len(error_df) > 0:
                valid_errors = error_df['combined_tip_carina_error_mm'].dropna()
                if len(valid_errors) > 0:
                    median_error = valid_errors.median()
                    hospital_simulation_errors[hospital].append(median_error)

    # Aggregate across simulations
    result_df = aggregate_simulation_stats(hospital_simulation_errors, use_median=True)

    print(f" Processed CL results for {len(result_df)} hospitals")
    return result_df


def process_ft_results_multi_sim(
    metadata_map: Dict,
    clinical_annotations: Dict,
    special_mappings: Dict,
    mappings: pd.DataFrame,
    hospital_weights: Dict,
    ft_dir: Optional[Path] = None,
    num_simulations: int = 10
) -> pd.DataFrame:
    """
    Load FT predictions, calculate distances, and aggregate statistics.

    Args:
        metadata_map: Image metadata from load_annotation_metadata()
        clinical_annotations: From load_all_clinical_annotations()
        special_mappings: From load_all_clinical_annotations()
        mappings: Hospital mappings DataFrame
        hospital_weights: Patient demographics dict
        ft_dir: Fine-tuning directory (defaults to DEFAULT_FT_DIR)
        num_simulations: Number of simulations (default: 10)

    Returns:
        DataFrame with aggregated statistics per hospital
    """
    print("\n Processing fine-tuning results...")

    # Load raw predictions
    hospital_simulations = load_ft_predictions_multi_sim(ft_dir, num_simulations)

    # Calculate errors for each hospital/simulation
    hospital_simulation_errors = {}

    for hospital, sim_predictions_list in hospital_simulations.items():
        hospital_simulation_errors[hospital] = []

        for sim_idx, predictions in enumerate(sim_predictions_list):
            # Calculate errors for this simulation
            error_df = calculate_errors_for_predictions(
                predictions, metadata_map, clinical_annotations,
                special_mappings, mappings, hospital_weights
            )

            # Calculate median combined error for this simulation
            if len(error_df) > 0:
                valid_errors = error_df['combined_tip_carina_error_mm'].dropna()
                if len(valid_errors) > 0:
                    median_error = valid_errors.median()
                    hospital_simulation_errors[hospital].append(median_error)

    # Aggregate across simulations
    result_df = aggregate_simulation_stats(hospital_simulation_errors, use_median=True)

    print(f" Processed FT results for {len(result_df)} hospitals")
    return result_df


def process_direct_results(
    metadata_map: Dict,
    clinical_annotations: Dict,
    special_mappings: Dict,
    mappings: pd.DataFrame,
    hospital_weights: Dict,
    direct_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load Direct predictions and calculate distances.

    Note: Direct predictions are from a single simulation, so no aggregation.

    Args:
        metadata_map: Image metadata from load_annotation_metadata()
        clinical_annotations: From load_all_clinical_annotations()
        special_mappings: From load_all_clinical_annotations()
        mappings: Hospital mappings DataFrame
        hospital_weights: Patient demographics dict
        direct_path: Direct predictions CSV path (defaults to DEFAULT_DIRECT_PATH)

    Returns:
        DataFrame with statistics per hospital (single simulation):
            - hospital_name
            - combined_error_median_mm
            - n_images
    """
    print("\n Processing direct prediction results...")

    # Load raw predictions
    predictions = load_direct_predictions(direct_path, mappings)

    # Calculate errors
    error_df = calculate_errors_for_predictions(
        predictions, metadata_map, clinical_annotations,
        special_mappings, mappings, hospital_weights
    )

    # Group by hospital and calculate median
    hospital_stats = []
    for hospital in error_df['hospital_name'].unique():
        hospital_errors = error_df[error_df['hospital_name'] == hospital]
        valid_errors = hospital_errors['combined_tip_carina_error_mm'].dropna()

        if len(valid_errors) > 0:
            hospital_stats.append({
                'hospital_name': hospital,
                'combined_error_median_mm': valid_errors.median(),
                'n_images': len(valid_errors)
            })

    result_df = pd.DataFrame(hospital_stats)

    print(f" Processed Direct results for {len(result_df)} hospitals")
    return result_df


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_hospital_display_name(normalized_name: str) -> str:
    """
    Convert normalized hospital name to proper display format.

    Args:
        normalized_name: Normalized hospital name (e.g., "University of Alberta")

    Returns:
        Display name (same format for consistency)
    """
    # This can be expanded with a mapping dict if needed
    # For now, just return the normalized name
    return normalized_name


def print_loading_summary(
    cl_df: Optional[pd.DataFrame] = None,
    ft_df: Optional[pd.DataFrame] = None,
    direct_df: Optional[pd.DataFrame] = None
):
    """
    Print a summary of loaded results.

    Args:
        cl_df: CL results DataFrame (optional)
        ft_df: FT results DataFrame (optional)
        direct_df: Direct results DataFrame (optional)
    """
    print("\n" + "="*80)
    print("LOADING SUMMARY")
    print("="*80)

    if cl_df is not None:
        print(f" Continual Learning: {len(cl_df)} hospitals")
        if 'n_simulations' in cl_df.columns:
            avg_sims = cl_df['n_simulations'].mean()
            print(f"  Average simulations per hospital: {avg_sims:.1f}/10")

    if ft_df is not None:
        print(f" Fine-tuning: {len(ft_df)} hospitals")
        if 'n_simulations' in ft_df.columns:
            avg_sims = ft_df['n_simulations'].mean()
            print(f"  Average simulations per hospital: {avg_sims:.1f}/10")

    if direct_df is not None:
        print(f" Direct Prediction: {len(direct_df)} hospitals")
        if 'n_images' in direct_df.columns:
            total_images = direct_df['n_images'].sum()
            print(f"  Total images: {total_images}")

    print("="*80)
