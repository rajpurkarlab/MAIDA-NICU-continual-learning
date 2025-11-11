#!/usr/bin/env python3
"""
Create eTable 5 with 10-simulation support and Alberta hospital included.

This version:
- Processes 10 simulations for both CL and FT
- Includes University of Alberta hospital (31 hospitals total)
- Plots average improvements in precision/recall from CL over FT/Direct
- Uses t-tests between CL and FT (paired across simulations)
- Uses one-sample t-test between CL and Direct (Direct has only 1 simulation)

Key updates from original:
- Handles new data structure with simulation_0 through simulation_9
- Processes CL data from 3 locations (VISH, EMMA, OISHI)
- Alberta annotations from latest_annotations directory
- Averages metrics across simulations before plotting improvements
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
import warnings
import re
from scipy import stats
from scipy.stats import ttest_ind, ttest_1samp
import argparse
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import requests

warnings.filterwarnings('ignore')

# Configure matplotlib for better rendering
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Bitstream Vera Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# Colors for methods (same as other scripts)
CL_COLOR = '#DC143C'  # Strong red (crimson) for continual learning
FT_COLOR = '#0066CC'  # Strong blue for fine-tuning

# Add parent directory to path to import utilities
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

# Import from standardized utility modules
from ett_width_utils import (
    load_all_clinical_annotations,
    load_annotation_metadata,
    load_hospital_weights_standardized,
    get_patient_weight_standardized,
    get_patient_gestational_age,
    get_ett_width_from_weight,
    get_ett_width_from_gestational_age,
    calculate_distance_with_ett_width_hierarchical,
    normalize_hospital_name,
    parse_coordinates
)

from prediction_loading_utils import (
    load_cl_predictions_multi_sim,
    load_ft_predictions_multi_sim,
    load_direct_predictions,
    CL_LOCATIONS as PREDICTION_LOADING_CL_LOCATIONS,  # Import CL locations from config
    DEFAULT_FT_DIR as PREDICTION_LOADING_FT_DIR,
    DEFAULT_DIRECT_PATH as PREDICTION_LOADING_DIRECT_PATH
)


# ==============================================================================
# CONFIGURATION - Repository Data Paths
# ==============================================================================

# Get repository root directory (2 levels up from this file: analysis/examples/)
REPO_ROOT = Path(__file__).parent.parent.parent

# Paths to data within the repository (relative to repo root)
DATA_DIR = REPO_ROOT / "data"
MAPPINGS_PATH = DATA_DIR / "mappings.csv"
DEMOGRAPHICS_DIR = DATA_DIR / "demographics"
ANNOTATIONS_DIR = DATA_DIR / "annotations" / "preprocessed_640x640"
TEST_ANNOTATIONS_PATH = ANNOTATIONS_DIR / "hospital-test-annotations.json"
TRAIN_ANNOTATIONS_PATH = ANNOTATIONS_DIR / "hospital-train-annotations.json"

# Default output directory for generated figures and tables
# UPDATE THIS PATH to your desired output location
DEFAULT_OUTPUT_DIR = Path("/path/to/output/etable5")  # UPDATE THIS PATH

# CL locations configuration is imported from prediction_loading_utils.py
# To customize CL experiment locations, update the CL_LOCATIONS variable in:
#   analysis/utils/prediction_loading_utils.py


# ==============================================================================
# HELPER FUNCTIONS (from missing create_etable4_forward_compatible module)
# ==============================================================================
# Note: The original script imported from create_etable4_forward_compatible.py
# If you need these functions, you may need to implement them or modify this script.
# Functions that were imported:
#   - get_proper_hospital_display_name()
#   - create_special_hospital_mapping()
#   - match_prediction_to_annotation()


# Hospital to country flag emoji mapping
HOSPITAL_COUNTRY_FLAGS = {
    'Alberta': '🇨🇦',
    'American-University-of-Beirut': '🇱🇧',
    'Childrens-Hospital-Colorado': '🇺🇸',
    'Chulalongkorn-University': '🇹🇭',
    'Dr-Sardjito-Hospital': '🇮🇩',
    'Fundacion-Santa-Fe-de-Bogota': '🇨🇴',
    'Indus': '🇵🇰',
    'International-Islamic-Medical-University': '🇲🇾',
    'Istanbul-Training-Research': '🇹🇷',
    'King-Abdulaziz-Hospital': '🇸🇦',
    'Kirikkale-Hospital': '🇹🇷',
    'Kirikkale-Hospital-': '🇹🇷',
    'La-Paz-University-Hospital': '🇪🇸',
    'Maharaj-Nakorn-Chiang-Mai-Hospital': '🇹🇭',
    'Medical-Center-of-South': '🇺🇸',
    'National-Cheng-Kung-University-Hospital': '🇹🇼',
    'National-University-Singapore': '🇸🇬',
    'Newark-Beth-Israel': '🇺🇸',
    'Osaka-Metropolitan-University': '🇯🇵',
    'Puerta-del-Mar-University-Hosptial': '🇪🇸',
    'SES': '🇨🇴',
    'Shiraz-University': '🇮🇷',
    'Sichuan-People-Hospital': '🇨🇳',
    'Sidra-Health': '🇶🇦',
    'Tel-Aviv-Medical-Center': '🇮🇱',
    'Tri-Service-General-Hospital': '🇹🇼',
    'Uni-Tubingen': '🇩🇪',
    'Universitaetsklinikum-Essen': '🇩🇪',
    'University-Hospital-Aachen': '🇩🇪',
    'University-of-Graz': '🇦🇹',
    'University-of-Kragujevac': '🇷🇸',
    'University-of-Linz': '🇦🇹',
}

# Cache directory for flag images
FLAGS_DIR = Path(__file__).parent / 'flags'
FLAGS_DIR.mkdir(exist_ok=True)

# Wrapper for compatibility with old code
def calculate_ett_carina_distance(tip_pred, carina_pred, image_id, test_images,
                                   clinical_annotations, mappings, hospital_weights, special_mapping):
    """Calculate predicted ETT-Carina distance using patient-specific ETT width calibration.

    This is a compatibility wrapper that uses calculate_distance_with_ett_width_hierarchical() from utils.
    """
    if tip_pred is None or carina_pred is None:
        return np.nan

    # Get file name with .png extension for metadata lookup
    file_name = image_id if image_id.endswith('.png') else f"{image_id}.png"

    # Get metadata from test_images (already loaded)
    metadata = test_images.get(image_id, {
        'orig_width': 640, 'orig_height': 640,
        'width': 640, 'height': 640
    })

    # Use the standardized distance calculation from utils
    # This handles ETT width hierarchical fallback automatically
    distance_mm = calculate_distance_with_ett_width_hierarchical(
        tip_pred, carina_pred, file_name, metadata,
        clinical_annotations, special_mapping, mappings, hospital_weights
    )

    # Convert mm to cm for compatibility with existing code
    if not np.isnan(distance_mm):
        return distance_mm / 10.0

    return np.nan


def hospital_name_matches(hospital1, hospital2):
    """Check if two hospital names match, accounting for different formats."""
    if hospital1 == hospital2:
        return True

    # Create a mapping of various hospital name formats
    hospital_equivalents = {
        'Alberta': ['Alberta', 'University-of-Alberta', 'University of Alberta'],
        'Indus': ['Indus', 'Indus-Hospital-and-Health-Network', 'Indus Hospital and Health Network'],
        'Kirikkale-Hospital-': ['Kirikkale-Hospital-', 'Kirikkale-Hospital', 'Kirikkale Hospital'],
        'Medical-Center-of-South': ['Medical-Center-of-South', 'Medical-University-of-South-Carolina', 'Medical University of South Carolina'],
        'National-University-Singapore': ['National-University-Singapore', 'National-University-of-Singapore', 'National University of Singapore'],
        'Newark-Beth-Israel': ['Newark-Beth-Israel', 'Newark-Beth-Israel-Medical-Center', 'Newark Beth Israel Medical Center'],
        'Osaka-Metropolitan-University': ['Osaka-Metropolitan-University', 'Osaka-Metropolitan-University-Hospital', 'Osaka Metropolitan University Hospital'],
        'Puerta-del-Mar-University-Hosptial': ['Puerta-del-Mar-University-Hosptial', 'Puerta-del-Mar-University-Hospital', 'Puerta del Mar University Hospital'],
        'SES': ['SES', 'SES-Hospital', 'Hospital SES'],
        'Shiraz-University': ['Shiraz-University', 'Shiraz-University-of-Medical-Sciences', 'Shiraz University of Medical Sciences'],
        'Sichuan-People-Hospital': ['Sichuan-People-Hospital', 'Sichuan-Provincial-People-Hospital', "Sichuan Provincial People's Hospital"],
        'Sidra-Health': ['Sidra-Health', 'Sidra-Medicine', 'Sidra Medicine'],
        'Uni-Tubingen': ['Uni-Tubingen', 'University-Hospital-of-Tuebingen', 'University Hospital of Tuebingen'],
        'Universitaetsklinikum-Essen': ['Universitaetsklinikum-Essen', 'University-of-Essen', 'University of Essen'],
        'University-of-Graz': ['University-of-Graz', 'Medical-University-of-Graz', 'Medical University of Graz'],
        'University-of-Kragujevac': ['University-of-Kragujevac', 'University-Clinical-Centre-Kragujevac', 'University Clinical Centre Kragujevac'],
        'University-of-Linz': ['University-of-Linz', 'Kepler-University-Hospital-Linz', 'Kepler University Hospital Linz']
    }

    # Check if both hospitals belong to the same equivalence group
    for key, equivalents in hospital_equivalents.items():
        if hospital1 in equivalents and hospital2 in equivalents:
            return True

    # Also check normalized versions
    norm1 = normalize_hospital_name(hospital1) if hospital1 else None
    norm2 = normalize_hospital_name(hospital2) if hospital2 else None

    if norm1 and norm2 and norm1 == norm2:
        return True

    return False

# Cache directory for flag images
FLAGS_DIR = Path(__file__).parent / 'flags'
FLAGS_DIR.mkdir(exist_ok=True)

def _flag_emoji_to_iso(flag_emoji):
    """Convert flag emoji to ISO 2-letter country code."""
    if len(flag_emoji) != 2:
        return None
    return ''.join([chr(ord(c) - 0x1F1E6 + 65) for c in flag_emoji]).lower()

def _download_flag_if_needed(iso_code: str, size: str = '48x36') -> Path:
    """Download PNG flag if not already present."""
    fname = f"{iso_code}_{size}.png"
    local_path = FLAGS_DIR / fname
    if local_path.exists():
        return local_path
    url = f"https://flagcdn.com/{size}/{iso_code}.png"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        with open(local_path, 'wb') as f:
            f.write(resp.content)
        return local_path
    except Exception:
        return None

def extract_ett_placement(annotation):
    """Extract ETT placement classification from annotation."""
    if 'classification' in annotation and 'attributes' in annotation['classification']:
        placement_value = annotation['classification']['attributes'].get('ETT placement')
        if isinstance(placement_value, list):
            return placement_value[0] if placement_value else None
        return placement_value
    return None

def extract_require_action(annotation):
    """Extract require_action flag from annotation."""
    if 'classification' in annotation and 'attributes' in annotation['classification']:
        require_action = annotation['classification']['attributes'].get('require_action')
        if isinstance(require_action, bool):
            return require_action
        elif isinstance(require_action, list):
            return require_action[0] if require_action else False
        elif isinstance(require_action, str):
            return require_action.lower() == 'true'
    return False

def calculate_precision_recall_at_k(ranked_samples, k, total_low):
    """Calculate precision@k and recall@k for detecting Low placement."""
    if len(ranked_samples) == 0 or total_low == 0:
        return 0.0, 0.0

    top_k = ranked_samples.head(min(k, len(ranked_samples)))

    # Count Low placements
    n_low = (top_k['ett_placement'] == 'Low').sum()

    precision = n_low / len(top_k)
    recall = n_low / total_low

    return precision, recall

def process_and_analyze_hospital(hospital_name, clinical_annotations, test_images, mappings, hospital_weights,
                                  special_mapping, cl_sim_predictions_list, ft_sim_predictions_list,
                                  direct_predictions, k_values=[5, 10, 15, 20]):
    """
    Process a single hospital's data across all simulations and calculate metrics for all K values.

    Returns:
        Dict with CL/FT/Direct precision and recall values for each simulation and K value
    """
    results = {
        'hospital': hospital_name,
        'total_low': 0,
        'total_test': 0
    }

    # Initialize storage for each K value
    for k in k_values:
        results[f'cl_precision_values_k{k}'] = []
        results[f'cl_recall_values_k{k}'] = []
        results[f'ft_precision_values_k{k}'] = []
        results[f'ft_recall_values_k{k}'] = []
        results[f'direct_precision_k{k}'] = None
        results[f'direct_recall_k{k}'] = None

    # First, collect all relevant test samples for this hospital
    test_samples = []

    # Process all image IDs to find this hospital's samples
    for image_id in test_images.keys():
        clinical_ann = clinical_annotations.get(image_id)

        if not clinical_ann:
            # Check special mapping for Essen, Osaka, Chiang Mai
            if special_mapping:
                ann_id = match_prediction_to_annotation(image_id, clinical_annotations, special_mapping)
                if ann_id:
                    clinical_ann = clinical_annotations.get(ann_id)

        if not clinical_ann:
            continue

        # Extract clinical labels
        ett_placement = extract_ett_placement(clinical_ann)
        if ett_placement is None:
            continue

        # Get hospital name for this sample
        sample_hospital = None

        # Check in direct predictions for hospital name
        if image_id in direct_predictions:
            sample_hospital = direct_predictions[image_id].get('hospital_name')
        # Check in first CL simulation
        elif cl_sim_predictions_list and len(cl_sim_predictions_list) > 0:
            if image_id in cl_sim_predictions_list[0]:
                sample_hospital = cl_sim_predictions_list[0][image_id].get('hospital_name')
        # Check in first FT simulation
        elif ft_sim_predictions_list and len(ft_sim_predictions_list) > 0:
            if image_id in ft_sim_predictions_list[0]:
                sample_hospital = ft_sim_predictions_list[0][image_id].get('hospital_name')

        # Skip if not our target hospital (use flexible matching)
        if not hospital_name_matches(sample_hospital, hospital_name):
            continue

        test_samples.append({
            'image_id': image_id,
            'ett_placement': ett_placement,
            'require_action': extract_require_action(clinical_ann)
        })

    # Also check for special hospital samples not in test_images
    special_prefixes = ['chiang_mai_p', 'chiang_p', 'osaka_p', 'essen_p', 'universitaetsklinikum_essen_p']
    # Add Alberta prefix
    if hospital_name == 'Alberta':
        special_prefixes.append('alberta_p')

    for prefix in special_prefixes:
        # Check in direct predictions
        for pred_id in direct_predictions.keys():
            if pred_id.startswith(prefix):
                ann_id = match_prediction_to_annotation(pred_id, clinical_annotations, special_mapping) if special_mapping else pred_id
                if not ann_id:
                    ann_id = pred_id

                if ann_id in clinical_annotations:
                    clinical_ann = clinical_annotations[ann_id]
                    ett_placement = extract_ett_placement(clinical_ann)

                    if ett_placement and pred_id not in [s['image_id'] for s in test_samples]:
                        pred_hospital = direct_predictions[pred_id].get('hospital_name')
                        if hospital_name_matches(pred_hospital, hospital_name):
                            test_samples.append({
                                'image_id': pred_id,
                                'ett_placement': ett_placement,
                                'require_action': extract_require_action(clinical_ann)
                            })

    if len(test_samples) == 0:
        return results

    # Calculate total Low placements
    results['total_test'] = len(test_samples)
    results['total_low'] = sum(1 for s in test_samples if s['ett_placement'] == 'Low')

    if results['total_low'] == 0:
        return results

    # Process CL simulations
    for sim_idx, cl_predictions in enumerate(cl_sim_predictions_list):
        if not cl_predictions:
            continue

        # Calculate distances for each sample
        sample_distances = []
        for sample in test_samples:
            image_id = sample['image_id']
            if image_id in cl_predictions:
                pred = cl_predictions[image_id]
                distance = calculate_ett_carina_distance(
                    pred.get('tip_pred'), pred.get('carina_pred'),
                    image_id, test_images, clinical_annotations, mappings,
                    hospital_weights, special_mapping
                )
                if not np.isnan(distance):
                    sample_distances.append({
                        'image_id': image_id,
                        'ett_placement': sample['ett_placement'],
                        'distance_cm': distance
                    })

        if len(sample_distances) >= 5:  # Need minimum samples for meaningful ranking
            # Rank by distance (ascending - smallest distance = most likely "too low")
            ranked_df = pd.DataFrame(sample_distances).sort_values('distance_cm')

            # Calculate metrics for each K value
            for k in k_values:
                prec, rec = calculate_precision_recall_at_k(ranked_df, k, results['total_low'])
                results[f'cl_precision_values_k{k}'].append(prec)
                results[f'cl_recall_values_k{k}'].append(rec)

    # Process FT simulations
    for ft_predictions in ft_sim_predictions_list:
        if not ft_predictions:
            continue

        sample_distances = []
        for sample in test_samples:
            image_id = sample['image_id']
            if image_id in ft_predictions:
                pred = ft_predictions[image_id]
                distance = calculate_ett_carina_distance(
                    pred.get('tip_pred'), pred.get('carina_pred'),
                    image_id, test_images, clinical_annotations, mappings,
                    hospital_weights, special_mapping
                )
                if not np.isnan(distance):
                    sample_distances.append({
                        'image_id': image_id,
                        'ett_placement': sample['ett_placement'],
                        'distance_cm': distance
                    })

        if len(sample_distances) >= 5:
            ranked_df = pd.DataFrame(sample_distances).sort_values('distance_cm')

            # Calculate metrics for each K value
            for k in k_values:
                prec, rec = calculate_precision_recall_at_k(ranked_df, k, results['total_low'])
                results[f'ft_precision_values_k{k}'].append(prec)
                results[f'ft_recall_values_k{k}'].append(rec)

    # Process Direct predictions (single simulation)
    if direct_predictions:
        sample_distances = []
        for sample in test_samples:
            image_id = sample['image_id']
            if image_id in direct_predictions:
                pred = direct_predictions[image_id]
                distance = calculate_ett_carina_distance(
                    pred.get('tip_pred'), pred.get('carina_pred'),
                    image_id, test_images, clinical_annotations, mappings,
                    hospital_weights, special_mapping
                )
                if not np.isnan(distance):
                    sample_distances.append({
                        'image_id': image_id,
                        'ett_placement': sample['ett_placement'],
                        'distance_cm': distance
                    })

        if len(sample_distances) >= 5:
            ranked_df = pd.DataFrame(sample_distances).sort_values('distance_cm')

            # Calculate metrics for each K value
            for k in k_values:
                prec, rec = calculate_precision_recall_at_k(ranked_df, k, results['total_low'])
                results[f'direct_precision_k{k}'] = prec
                results[f'direct_recall_k{k}'] = rec

    return results

def create_improvement_plot(hospital_results, output_dir, k=10):
    """
    Create horizontal bar plot showing precision and recall improvements for a specific K value.
    CL average - FT average and CL average - Direct.
    """

    # Prepare data for plotting
    plot_data = []

    for _, row in hospital_results.iterrows():
        hospital = row['hospital']

        # Calculate CL averages for this K
        cl_prec_mean = np.mean(row[f'cl_precision_values_k{k}']) * 100 if row[f'cl_precision_values_k{k}'] else 0
        cl_rec_mean = np.mean(row[f'cl_recall_values_k{k}']) * 100 if row[f'cl_recall_values_k{k}'] else 0

        # Calculate FT averages for this K
        ft_prec_mean = np.mean(row[f'ft_precision_values_k{k}']) * 100 if row[f'ft_precision_values_k{k}'] else 0
        ft_rec_mean = np.mean(row[f'ft_recall_values_k{k}']) * 100 if row[f'ft_recall_values_k{k}'] else 0

        # Direct values for this K
        direct_prec = row[f'direct_precision_k{k}'] * 100 if row[f'direct_precision_k{k}'] is not None else 0
        direct_rec = row[f'direct_recall_k{k}'] * 100 if row[f'direct_recall_k{k}'] is not None else 0

        # Calculate improvements
        ft_prec_improvement = cl_prec_mean - ft_prec_mean
        ft_rec_improvement = cl_rec_mean - ft_rec_mean
        direct_prec_improvement = cl_prec_mean - direct_prec
        direct_rec_improvement = cl_rec_mean - direct_rec

        plot_data.append({
            'hospital': hospital,
            'ft_prec_improvement': ft_prec_improvement,
            'ft_rec_improvement': ft_rec_improvement,
            'ft_total_improvement': ft_prec_improvement + ft_rec_improvement,
            'direct_prec_improvement': direct_prec_improvement,
            'direct_rec_improvement': direct_rec_improvement,
            'direct_total_improvement': direct_prec_improvement + direct_rec_improvement,
            'total_low': row['total_low'],
            'total_test': row['total_test']
        })

    plot_df = pd.DataFrame(plot_data)

    # Create two plots: CL vs FT and CL vs Direct
    for comparison in ['ft', 'direct']:
        if comparison == 'ft':
            prec_col = 'ft_prec_improvement'
            rec_col = 'ft_rec_improvement'
            total_col = 'ft_total_improvement'
            title_suffix = 'vs Fine-tuning'
            filename_suffix = 'vs_ft'
        else:
            prec_col = 'direct_prec_improvement'
            rec_col = 'direct_rec_improvement'
            total_col = 'direct_total_improvement'
            title_suffix = 'vs Direct Prediction'
            filename_suffix = 'vs_direct'

        # Sort by total improvement
        plot_df = plot_df.sort_values(total_col, ascending=True).reset_index(drop=True)

        # Create figure
        fig_height = max(12, len(plot_df) * 0.4)
        fig, ax = plt.subplots(figsize=(12, fig_height))

        hospitals = plot_df['hospital'].values
        precision_improvements = plot_df[prec_col].values
        recall_improvements = plot_df[rec_col].values
        total_low = plot_df['total_low'].values
        total_test = plot_df['total_test'].values

        # Create horizontal bars
        y = np.arange(len(hospitals)) * 1.2
        height = 0.85

        for i, (prec_imp, rec_imp) in enumerate(zip(precision_improvements, recall_improvements)):
            # Recall bar (blue)
            ax.barh(y[i], rec_imp, height, color=FT_COLOR, alpha=0.85,
                   edgecolor='white', linewidth=1.5)

            # Precision bar (red) - stacked
            ax.barh(y[i], prec_imp, height, left=rec_imp, color=CL_COLOR,
                   alpha=0.85, edgecolor='white', linewidth=1.5)

        # Add vertical line at x=0
        ax.axvline(x=0, color='black', linewidth=1.5, zorder=5)

        # Set y-axis labels
        hospital_labels = []
        for hosp, n_low, n_total in zip(hospitals, total_low, total_test):
            display_name = get_proper_hospital_display_name(hosp)
            hospital_labels.append(f'{display_name} (low: {n_low}/{n_total})')

        ax.set_yticks(y)
        ax.set_yticklabels(hospital_labels, fontsize=10)

        # Move y-axis labels left for flags
        ax.tick_params(axis='y', pad=30)

        # Add flags
        for i, hospital in enumerate(hospitals):
            flag_emoji = HOSPITAL_COUNTRY_FLAGS.get(hospital)
            if not flag_emoji:
                continue
            iso_code = _flag_emoji_to_iso(flag_emoji)
            if not iso_code:
                continue
            flag_path = _download_flag_if_needed(iso_code)
            if not flag_path or not flag_path.exists():
                continue
            try:
                img = plt.imread(str(flag_path))
                imagebox = OffsetImage(img, zoom=0.35)
                trans = ax.get_yaxis_transform()
                ab = AnnotationBbox(imagebox, (-0.02, i * 1.2),
                                  xycoords=trans, frameon=False,
                                  box_alignment=(1, 0.5), pad=0)
                ax.add_artist(ab)
            except Exception:
                pass

        # Styling (removed title as requested)
        ax.set_xlabel(f'Improvement in Precision and Recall @ {k} (%)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Hospital', fontsize=13, fontweight='bold')

        ax.grid(False)
        ax.set_axisbelow(True)

        # Set x-axis limits
        x_max = max(precision_improvements + recall_improvements) + 5
        x_min = min(precision_improvements + recall_improvements) - 5
        x_max = max(x_max, 5)
        x_min = min(x_min, -5)
        ax.set_xlim(x_min, x_max)

        # Set y-axis limits
        ax.set_ylim(-0.8, y[-1] + 0.8)

        # Add background shading
        ax.axvspan(0, x_max, alpha=0.03, color='green', zorder=0)
        ax.axvspan(x_min, 0, alpha=0.03, color='red', zorder=0)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=CL_COLOR, alpha=0.85, label=f'Precision@{k} improvement'),
            Patch(facecolor=FT_COLOR, alpha=0.85, label=f'Recall@{k} improvement')
        ]

        ax.legend(handles=legend_elements, loc='lower right', fontsize=11,
                 frameon=True, fancybox=True, shadow=False, framealpha=0.95)

        # Spine styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)

        plt.tight_layout()
        plt.subplots_adjust(left=0.25)

        # Save with K value in filename
        output_path = output_dir / f'etable5_multi_sim_improvements_k{k}_{filename_suffix}.png'
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        plt.close()

        print(f"    Saved improvement plot: {output_path}")

def get_formal_hospital_name(hospital_key):
    """Get formal hospital name for alphabetical sorting."""
    formal_names = {
        'Alberta': 'University of Alberta',
        'American-University-of-Beirut': 'American University of Beirut',
        'Childrens-Hospital-Colorado': "Children's Hospital Colorado",
        'Chulalongkorn-University': 'Chulalongkorn University',
        'Dr-Sardjito-Hospital': 'Dr. Sardjito Hospital',
        'Fundacion-Santa-Fe-de-Bogota': 'Fundación Santa Fe de Bogotá',
        'Indus': 'Indus Hospital and Health Network',
        'International-Islamic-Medical-University': 'International Islamic Medical University',
        'Istanbul-Training-Research': 'Istanbul Training and Research Hospital',
        'King-Abdulaziz-Hospital': 'King Abdulaziz Hospital',
        'Kirikkale-Hospital': 'Kirikkale Hospital',
        'Kirikkale-Hospital-': 'Kirikkale Hospital',
        'La-Paz-University-Hospital': 'La Paz University Hospital',
        'Maharaj-Nakorn-Chiang-Mai-Hospital': 'Maharaj Nakorn Chiang Mai Hospital',
        'Medical-Center-of-South': 'Medical University of South Carolina',
        'National-Cheng-Kung-University-Hospital': 'National Cheng Kung University Hospital',
        'National-University-Singapore': 'National University of Singapore',
        'Newark-Beth-Israel': 'Newark Beth Israel Medical Center',
        'Osaka-Metropolitan-University': 'Osaka Metropolitan University Hospital',
        'Puerta-del-Mar-University-Hosptial': 'Puerta del Mar University Hospital',
        'SES': 'Hospital SES',
        'Shiraz-University': 'Shiraz University of Medical Sciences',
        'Sichuan-People-Hospital': "Sichuan Provincial People's Hospital",
        'Sidra-Health': 'Sidra Medicine',
        'Tel-Aviv-Medical-Center': 'Tel Aviv Medical Center',
        'Tri-Service-General-Hospital': 'Tri-Service General Hospital',
        'Uni-Tubingen': 'University Hospital of Tuebingen',
        'Universitaetsklinikum-Essen': 'University of Essen',
        'University-Hospital-Aachen': 'University Hospital Aachen',
        'University-of-Graz': 'Medical University of Graz',
        'University-of-Kragujevac': 'University Clinical Centre Kragujevac',
        'University-of-Linz': 'Kepler University Hospital Linz',
    }
    return formal_names.get(hospital_key, hospital_key)

def calculate_95_ci(values):
    """Calculate mean and 95% confidence interval."""
    if len(values) == 0:
        return 0, 0, 0

    values = np.array(values)
    mean = np.mean(values)

    if len(values) == 1:
        return mean, mean, mean

    # Calculate 95% CI using t-distribution
    from scipy import stats
    confidence = 0.95
    n = len(values)
    std_err = stats.sem(values)
    interval = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)

    return mean, mean - interval, mean + interval

def get_significance_stars(p_value):
    """Get significance stars based on p-value."""
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return ''

def create_formatted_table(results_df, output_dir):
    """Create formatted table matching the sample format."""

    # Sort hospitals by formal name
    results_df['formal_name'] = results_df['hospital'].apply(get_formal_hospital_name)
    results_df = results_df.sort_values('formal_name')

    # Initialize the table data
    table_rows = []

    for _, row in results_df.iterrows():
        hospital_formal = row['formal_name']

        # Process each K value (5 and 15)
        prec5_cl_values = row['cl_precision_values_k5']
        prec15_cl_values = row['cl_precision_values_k15']
        prec5_ft_values = row['ft_precision_values_k5']
        prec15_ft_values = row['ft_precision_values_k15']
        prec5_direct = row['direct_precision_k5']
        prec15_direct = row['direct_precision_k15']

        rec5_cl_values = row['cl_recall_values_k5']
        rec15_cl_values = row['cl_recall_values_k15']
        rec5_ft_values = row['ft_recall_values_k5']
        rec15_ft_values = row['ft_recall_values_k15']
        rec5_direct = row['direct_recall_k5']
        rec15_direct = row['direct_recall_k15']

        # Calculate means and CIs for CL
        prec5_cl_mean, prec5_cl_low, prec5_cl_high = calculate_95_ci(prec5_cl_values)
        prec15_cl_mean, prec15_cl_low, prec15_cl_high = calculate_95_ci(prec15_cl_values)
        rec5_cl_mean, rec5_cl_low, rec5_cl_high = calculate_95_ci(rec5_cl_values)
        rec15_cl_mean, rec15_cl_low, rec15_cl_high = calculate_95_ci(rec15_cl_values)

        # Calculate means and CIs for FT (FT also has 10 simulations)
        prec5_ft_mean, prec5_ft_low, prec5_ft_high = calculate_95_ci(prec5_ft_values)
        prec15_ft_mean, prec15_ft_low, prec15_ft_high = calculate_95_ci(prec15_ft_values)
        rec5_ft_mean, rec5_ft_low, rec5_ft_high = calculate_95_ci(rec5_ft_values)
        rec15_ft_mean, rec15_ft_low, rec15_ft_high = calculate_95_ci(rec15_ft_values)

        # Get significance stars for K=15 ONLY (only if CL > comparison)
        prec15_cl_ft_stars = ''
        prec15_cl_direct_stars = ''
        rec15_cl_ft_stars = ''
        rec15_cl_direct_stars = ''

        # Check if CL > FT for precision@15
        if prec15_cl_mean > prec15_ft_mean:
            p15 = row.get('cl_vs_ft_prec_p_k15', 1.0)
            prec15_cl_ft_stars = get_significance_stars(p15)

        # Check if CL > Direct for precision@15
        if prec15_direct is not None and prec15_cl_mean > prec15_direct:
            p15 = row.get('cl_vs_direct_prec_p_k15', 1.0)
            prec15_cl_direct_stars = get_significance_stars(p15)

        # Check if CL > FT for recall@15
        if rec15_cl_mean > rec15_ft_mean:
            p15 = row.get('cl_vs_ft_rec_p_k15', 1.0)
            rec15_cl_ft_stars = get_significance_stars(p15)

        # Check if CL > Direct for recall@15
        if rec15_direct is not None and rec15_cl_mean > rec15_direct:
            p15 = row.get('cl_vs_direct_rec_p_k15', 1.0)
            rec15_cl_direct_stars = get_significance_stars(p15)

        # Create rows for this hospital
        # Continual Learning row
        cl_row = {
            'Hospital': hospital_formal,
            'Method': 'Continual Learning',
            'Precision@5 (%)': f"{prec5_cl_mean*100:.1f} ({prec5_cl_low*100:.1f}-{prec5_cl_high*100:.1f})",
            'Precision@15 (%)': f"{prec15_cl_mean*100:.1f} ({prec15_cl_low*100:.1f}-{prec15_cl_high*100:.1f})",
            'CL > FT (Prec@15)': prec15_cl_ft_stars,
            'CL > Direct (Prec@15)': prec15_cl_direct_stars,
            'Recall@5 (%)': f"{rec5_cl_mean*100:.1f} ({rec5_cl_low*100:.1f}-{rec5_cl_high*100:.1f})",
            'Recall@15 (%)': f"{rec15_cl_mean*100:.1f} ({rec15_cl_low*100:.1f}-{rec15_cl_high*100:.1f})",
            'CL > FT (Rec@15)': rec15_cl_ft_stars,
            'CL > Direct (Rec@15)': rec15_cl_direct_stars
        }

        # Fine-tuning row (now with 95% CIs)
        ft_row = {
            'Hospital': '',  # Empty for second row
            'Method': 'Fine-tuning',
            'Precision@5 (%)': f"{prec5_ft_mean*100:.1f} ({prec5_ft_low*100:.1f}-{prec5_ft_high*100:.1f})" if prec5_ft_values else '',
            'Precision@15 (%)': f"{prec15_ft_mean*100:.1f} ({prec15_ft_low*100:.1f}-{prec15_ft_high*100:.1f})" if prec15_ft_values else '',
            'CL > FT (Prec@15)': '',
            'CL > Direct (Prec@15)': '',
            'Recall@5 (%)': f"{rec5_ft_mean*100:.1f} ({rec5_ft_low*100:.1f}-{rec5_ft_high*100:.1f})" if rec5_ft_values else '',
            'Recall@15 (%)': f"{rec15_ft_mean*100:.1f} ({rec15_ft_low*100:.1f}-{rec15_ft_high*100:.1f})" if rec15_ft_values else '',
            'CL > FT (Rec@15)': '',
            'CL > Direct (Rec@15)': ''
        }

        # Direct Prediction row
        direct_row = {
            'Hospital': '',  # Empty for third row
            'Method': 'Direct Prediction',
            'Precision@5 (%)': f"{prec5_direct*100:.1f}" if prec5_direct is not None else '',
            'Precision@15 (%)': f"{prec15_direct*100:.1f}" if prec15_direct is not None else '',
            'CL > FT (Prec@15)': '',
            'CL > Direct (Prec@15)': '',
            'Recall@5 (%)': f"{rec5_direct*100:.1f}" if rec5_direct is not None else '',
            'Recall@15 (%)': f"{rec15_direct*100:.1f}" if rec15_direct is not None else '',
            'CL > FT (Rec@15)': '',
            'CL > Direct (Rec@15)': ''
        }

        table_rows.extend([cl_row, ft_row, direct_row])

    # Create DataFrame
    table_df = pd.DataFrame(table_rows)

    # Column names are already set in the dictionaries above - no renaming needed
    # Columns: Hospital, Method, Precision@5 (%), Precision@15 (%),
    #          CL > FT (Prec@15), CL > Direct (Prec@15), Recall@5 (%), Recall@15 (%),
    #          CL > FT (Rec@15), CL > Direct (Rec@15)

    # Save to CSV
    table_path = output_dir / 'etable_5_formatted.csv'
    table_df.to_csv(table_path, index=False)

    print(f"    Saved formatted table: {table_path}")

    return table_df

def perform_statistical_tests_for_k(hospital_results, k):
    """
    Perform statistical tests for a specific K value:
    - Independent t-test for CL vs FT (treating simulations as independent samples)
    - One-sample t-test for CL vs Direct (Direct has only 1 value)
    """

    for idx, row in hospital_results.iterrows():
        hospital = row['hospital']

        # Get values for this K
        cl_prec_values = np.array(row[f'cl_precision_values_k{k}'])
        cl_rec_values = np.array(row[f'cl_recall_values_k{k}'])
        ft_prec_values = np.array(row[f'ft_precision_values_k{k}'])
        ft_rec_values = np.array(row[f'ft_recall_values_k{k}'])
        direct_prec = row[f'direct_precision_k{k}']
        direct_rec = row[f'direct_recall_k{k}']

        # CL vs FT tests (independent t-test treating simulations as independent)
        if len(cl_prec_values) > 1 and len(ft_prec_values) > 1:
            try:
                # Use independent samples t-test
                _, p_prec = ttest_ind(cl_prec_values, ft_prec_values)
                _, p_rec = ttest_ind(cl_rec_values, ft_rec_values)
                hospital_results.at[idx, f'cl_vs_ft_prec_p_k{k}'] = p_prec
                hospital_results.at[idx, f'cl_vs_ft_rec_p_k{k}'] = p_rec
            except:
                hospital_results.at[idx, f'cl_vs_ft_prec_p_k{k}'] = np.nan
                hospital_results.at[idx, f'cl_vs_ft_rec_p_k{k}'] = np.nan
        else:
            hospital_results.at[idx, f'cl_vs_ft_prec_p_k{k}'] = np.nan
            hospital_results.at[idx, f'cl_vs_ft_rec_p_k{k}'] = np.nan

        # CL vs Direct tests (one-sample t-test)
        if len(cl_prec_values) > 1 and direct_prec is not None:
            try:
                _, p_prec = ttest_1samp(cl_prec_values, direct_prec)
                _, p_rec = ttest_1samp(cl_rec_values, direct_rec)
                hospital_results.at[idx, f'cl_vs_direct_prec_p_k{k}'] = p_prec
                hospital_results.at[idx, f'cl_vs_direct_rec_p_k{k}'] = p_rec
            except:
                hospital_results.at[idx, f'cl_vs_direct_prec_p_k{k}'] = np.nan
                hospital_results.at[idx, f'cl_vs_direct_rec_p_k{k}'] = np.nan
        else:
            hospital_results.at[idx, f'cl_vs_direct_prec_p_k{k}'] = np.nan
            hospital_results.at[idx, f'cl_vs_direct_rec_p_k{k}'] = np.nan

    return hospital_results

def main():
    """Main function to create eTable 5 with multi-simulation support."""

    parser = argparse.ArgumentParser(description="Create eTable 5 with 10-simulation support and Alberta")
    parser.add_argument("--output-dir", type=str,
                       default=str(DEFAULT_OUTPUT_DIR),
                       help="Output directory for results")
    args = parser.parse_args()

    # Use output directory from configuration or command line argument
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("CREATING eTABLE 5 WITH 10-SIMULATION SUPPORT AND ALBERTA")
    print("="*70)
    print(f" Output directory: {output_dir}")
    print(f" Processing K values: [5, 15] for precision and recall metrics")
    print("="*70)

    # Use CL locations from prediction_loading_utils.py configuration
    # This ensures consistency across all analysis scripts
    locations = PREDICTION_LOADING_CL_LOCATIONS

    # Mapping from simplified directory names to the names used for FT files
    hospital_name_mapping_for_ft = {
        'Alberta': 'Alberta',
        'American-University-of-Beirut': 'American-University-of-Beirut',
        'Childrens-Hospital-Colorado': 'Childrens-Hospital-Colorado',
        'Chulalongkorn-University': 'Chulalongkorn-University',
        'Dr-Sardjito-Hospital': 'Dr-Sardjito-Hospital',
        'Fundacion-Santa-Fe-de-Bogota': 'Fundacion-Santa-Fe-de-Bogota',
        'Indus': 'Indus',
        'International-Islamic-Medical-University': 'International-Islamic-Medical-University',
        'Istanbul-Training-Research': 'Istanbul-Training-Research',
        'King-Abdulaziz-Hospital': 'King-Abdulaziz-Hospital',
        'Kirikkale-Hospital-': 'Kirikkale-Hospital-',
        'Sichuan-People-Hospital': 'Sichuan-People-Hospital',
        'Sidra-Health': 'Sidra-Health',
        'Tel-Aviv-Medical-Center': 'Tel-Aviv-Medical-Center',
        'Tri-Service-General-Hospital': 'Tri-Service-General-Hospital',
        'Uni-Tubingen': 'Uni-Tubingen',
        'Universitaetsklinikum-Essen': 'Universitaetsklinikum-Essen',
        'University-Hospital-Aachen': 'University-Hospital-Aachen',
        'University-of-Graz': 'University-of-Graz',
        'University-of-Kragujevac': 'University-of-Kragujevac',
        'University-of-Linz': 'University-of-Linz',
        'La-Paz-University-Hospital': 'La-Paz-University-Hospital',
        'Maharaj-Nakorn-Chiang-Mai-Hospital': 'Maharaj-Nakorn-Chiang-Mai-Hospital',
        'Medical-Center-of-South': 'Medical-Center-of-South',
        'National-Cheng-Kung-University-Hospital': 'National-Cheng-Kung-University-Hospital',
        'National-University-Singapore': 'National-University-Singapore',
        'Newark-Beth-Israel': 'Newark-Beth-Israel',
        'Osaka-Metropolitan-University': 'Osaka-Metropolitan-University',
        'Puerta-del-Mar-University-Hosptial': 'Puerta-del-Mar-University-Hosptial',
        'SES': 'SES',
        'Shiraz-University': 'Shiraz-University'
    }

    # Load data
    print("\n Loading data...")

    # Use paths from configuration
    print("   Loading mappings and patient weights...")
    mappings = pd.read_csv(MAPPINGS_PATH)
    hospital_weights = load_hospital_weights_standardized(DEMOGRAPHICS_DIR)

    print("   Loading clinical annotations (including Alberta)...")
    clinical_annotations, _ = load_all_clinical_annotations()

    print("   Loading test image metadata...")
    test_images = {}
    metadata_map = load_annotation_metadata(TEST_ANNOTATIONS_PATH)
    with open(TEST_ANNOTATIONS_PATH, 'r') as f:
        test_data = json.load(f)
        for img in test_data['images']:
            file_name = img['file_name'].replace('.png', '')
            test_images[file_name] = metadata_map.get(img['file_name'], {})

    print(f"    Loaded {len(mappings)} ID mappings")
    print(f"    Loaded weight data for {len(hospital_weights)} hospitals")
    print(f"    Loaded {len(clinical_annotations)} clinical annotations")
    print(f"    Loaded {len(test_images)} test images")

    # Create special hospital mapping in the old format for compatibility with match_prediction_to_annotation()
    print("   Creating special hospital mapping...")
    special_mapping = create_special_hospital_mapping(clinical_annotations)
    print(f"    Created {len(special_mapping)} special hospital mappings")

    # Load CL predictions from all simulations (using standardized utils)
    print("\n Loading CL holdout predictions from 10 simulations...")
    cl_hospital_simulations = load_cl_predictions_multi_sim(locations=locations, num_simulations=10)
    print(f"    Loaded CL data for {len(cl_hospital_simulations)} hospitals")

    # Load FT predictions from all simulations (using standardized utils)
    print("\n Loading fine-tuning predictions from 10 simulations...")
    ft_dir = "/home/vir247/scratch/nicu/MAIDA-continual-learning/outputs/fine_tune/pretrained/target-hospital-only"
    ft_simulations = load_ft_predictions_multi_sim(ft_dir=ft_dir, num_simulations=10)
    print(f"    Loaded FT data for {len(ft_simulations)} hospitals")

    # Load Direct predictions (single simulation, using standardized utils)
    print("\n Loading direct predictions...")
    direct_path = "/home/vir247/scratch/nicu/MAIDA-continual-learning/outputs/nicu_inference/simulation_0/nicu_test_predictions.csv"
    direct_predictions = load_direct_predictions(direct_path=direct_path, mappings=mappings)
    print(f"    Loaded {len(direct_predictions)} direct predictions")

    # Define K values to analyze (only 5 and 15 for final table)
    k_values = [5, 15]  # Changed to only analyze K=5 and K=15 for the table

    # Process each hospital
    print(f"\n Analyzing performance for each hospital at K values: {k_values}...")

    results = []

    # Get all unique hospitals (excluding New Somerset)
    all_hospitals = set()
    for hospital_list in cl_hospital_simulations.keys():
        all_hospitals.add(hospital_list)

    # Remove New Somerset if present
    all_hospitals = {h for h in all_hospitals if 'Somerset' not in h}

    print(f"   Processing {len(all_hospitals)} hospitals (excluding New Somerset)...")

    for hospital_name in sorted(all_hospitals):
        # Get CL simulations for this hospital
        cl_sim_list = cl_hospital_simulations.get(hospital_name, [])

        # Get FT simulations for this hospital
        ft_sim_list = ft_simulations.get(hospital_name, [])


        # Process hospital for all K values
        hospital_results = process_and_analyze_hospital(
            hospital_name, clinical_annotations, test_images, mappings, hospital_weights,
            special_mapping, cl_sim_list, ft_sim_list, direct_predictions, k_values=k_values
        )

        if hospital_results['total_low'] > 0:
            results.append(hospital_results)

            # Print progress (using K=5 for summary)
            cl_sims = len(hospital_results.get('cl_precision_values_k5', []))
            ft_sims = len(hospital_results.get('ft_precision_values_k5', []))
            print(f"    {get_proper_hospital_display_name(hospital_name)}: "
                  f"{cl_sims} CL sims, {ft_sims} FT sims, "
                  f"{hospital_results['total_low']} low cases")

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    print(f"\n Processed {len(results_df)} hospitals with sufficient data")

    # Perform statistical tests for each K value
    print("\n Performing statistical tests for each K value...")
    for k in k_values:
        print(f"   Testing K={k}...")
        results_df = perform_statistical_tests_for_k(results_df, k)

    # Save detailed results
    results_df.to_csv(output_dir / f'etable5_multi_sim_detailed_all_k.csv', index=False)
    print(f"    Saved detailed results to etable5_multi_sim_detailed_all_k.csv")

    # Create formatted table matching the sample format
    print("\n Creating formatted table...")
    formatted_table = create_formatted_table(results_df, output_dir)

    # Create improvement plots for each K value
    print("\n Creating improvement visualizations for each K value...")
    for k in k_values:
        print(f"   Creating plots for K={k}...")
        create_improvement_plot(results_df, output_dir, k=k)

    # Create summary statistics for each K value
    print("\n Summary Statistics by K Value:")

    for k in k_values:
        print(f"\n   === K = {k} ===")

        # Calculate overall averages for this K
        overall_cl_prec = []
        overall_cl_rec = []
        overall_ft_prec = []
        overall_ft_rec = []
        overall_direct_prec = []
        overall_direct_rec = []

        for _, row in results_df.iterrows():
            if row[f'cl_precision_values_k{k}']:
                overall_cl_prec.extend(row[f'cl_precision_values_k{k}'])
                overall_cl_rec.extend(row[f'cl_recall_values_k{k}'])
            if row[f'ft_precision_values_k{k}']:
                overall_ft_prec.extend(row[f'ft_precision_values_k{k}'])
                overall_ft_rec.extend(row[f'ft_recall_values_k{k}'])
            if row[f'direct_precision_k{k}'] is not None:
                overall_direct_prec.append(row[f'direct_precision_k{k}'])
                overall_direct_rec.append(row[f'direct_recall_k{k}'])

        print(f"   Continual Learning (CL):")
        print(f"      Precision@{k}: {np.mean(overall_cl_prec)*100:.1f}% ± {np.std(overall_cl_prec)*100:.1f}%")
        print(f"      Recall@{k}: {np.mean(overall_cl_rec)*100:.1f}% ± {np.std(overall_cl_rec)*100:.1f}%")

        print(f"   Fine-tuning (FT):")
        print(f"      Precision@{k}: {np.mean(overall_ft_prec)*100:.1f}% ± {np.std(overall_ft_prec)*100:.1f}%")
        print(f"      Recall@{k}: {np.mean(overall_ft_rec)*100:.1f}% ± {np.std(overall_ft_rec)*100:.1f}%")

        print(f"   Direct Prediction:")
        print(f"      Precision@{k}: {np.mean(overall_direct_prec)*100:.1f}%")
        print(f"      Recall@{k}: {np.mean(overall_direct_rec)*100:.1f}%")

        # Count significant differences for this K
        sig_cl_ft_prec = (results_df[f'cl_vs_ft_prec_p_k{k}'] < 0.05).sum()
        sig_cl_ft_rec = (results_df[f'cl_vs_ft_rec_p_k{k}'] < 0.05).sum()
        sig_cl_direct_prec = (results_df[f'cl_vs_direct_prec_p_k{k}'] < 0.05).sum()
        sig_cl_direct_rec = (results_df[f'cl_vs_direct_rec_p_k{k}'] < 0.05).sum()

        print(f"   Statistical Significance (p<0.05):")
        print(f"      CL vs FT Precision: {sig_cl_ft_prec}/{len(results_df)} hospitals")
        print(f"      CL vs FT Recall: {sig_cl_ft_rec}/{len(results_df)} hospitals")
        print(f"      CL vs Direct Precision: {sig_cl_direct_prec}/{len(results_df)} hospitals")
        print(f"      CL vs Direct Recall: {sig_cl_direct_rec}/{len(results_df)} hospitals")

    print("\n Analysis complete!")
    print(f"\nGenerated files in {output_dir}:")
    print(f"  - etable5_multi_sim_detailed_all_k.csv")
    print(f"  - etable_5_formatted.csv (formatted table matching sample)")
    for k in k_values:
        print(f"  - etable5_multi_sim_improvements_k{k}_vs_ft.png")
        print(f"  - etable5_multi_sim_improvements_k{k}_vs_direct.png")

if __name__ == "__main__":
    main()