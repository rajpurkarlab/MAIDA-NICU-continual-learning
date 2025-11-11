#!/usr/bin/env python3
"""
Create eTable 4: Comparison of Continual Learning vs Fine-tuning vs Direct Prediction

Uses standardized utility functions following test_prediction_loading_utils.py patterns.

Statistical tests:
- CL vs FT: Two-sample t-test (both have 10 simulations)
- CL vs Direct: One-sample t-test (CL distribution vs Direct point estimate)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats
import argparse
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.lines import Line2D
import requests
import sys

# Add parent directory to path to import utilities
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

# Import standardized utilities
from ett_width_utils import (
    load_all_clinical_annotations,
    load_annotation_metadata,
    load_hospital_weights_standardized,
    normalize_hospital_name
)

from prediction_loading_utils import (
    load_cl_predictions_multi_sim,
    load_ft_predictions_multi_sim,
    calculate_errors_for_predictions,
    aggregate_simulation_stats
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
ANNOTATIONS_PATH = DATA_DIR / "annotations" / "preprocessed_640x640" / "hospital-test-annotations.json"

# Output directory for generated figures and tables
# UPDATE THIS PATH to your desired output location
OUTPUT_DIR = Path("/path/to/output/etable4")  # UPDATE THIS PATH

# Formal hospital names mapping (from normalized data format to CLAUDE.md formal names)
FORMAL_HOSPITAL_NAMES = {
    'Alberta': 'University of Alberta',
    'American-University-of-Beirut': 'American University of Beirut',
    'Childrens-Hospital-Colorado': "Children's Hospital Colorado",
    'Chulalongkorn-University': 'Chulalongkorn University',
    'Dr-Sardjito-Hospital': 'Dr Sardjito Hospital',
    'Fundacion-Santa-Fe-de-Bogota': 'Fundacion Santa Fe de Bogota',
    'Indus': 'Indus Hospital and Health Network',
    'International-Islamic-Medical-University': 'International Islamic Medical University',
    'Istanbul-Training-Research': 'Istanbul Training and Research Hospital',
    'King-Abdulaziz-Hospital': 'King Abdulaziz Hospital',
    'Kirikkale-Hospital': 'Kirikkale Hospital',
    'La-Paz-University-Hospital': 'La Paz University Hospital',
    'Maharaj-Nakorn-Chiang-Mai-Hospital': 'Maharaj Nakorn Chiang Mai Hospital',
    'Medical-Center-of-South': 'Medical University of South Carolina',
    'National-Cheng-Kung-University-Hospital': 'National Cheng Kung University Hospital',
    'National-University-Singapore': 'National University Singapore',
    'Newark-Beth-Israel': 'Newark Beth Israel',
    'New-Somerset-Hospital': 'New Somerset Hospital',
    'Osaka-Metropolitan-University': 'Osaka Metropolitan University',
    'Puerta-del-Mar-University-Hosptial': 'Puerta del Mar University Hospital',
    'SES': 'SES Hospital',
    'Shiraz-University': 'Shiraz University',
    'Sichuan-People-Hospital': "Sichuan Provincial People's Hospital",
    'Sidra-Health': 'Sidra Medicine',
    'Tel-Aviv-Medical-Center': 'Tel Aviv Medical Center',
    'Tri-Service-General-Hospital': 'Tri Service General Hospital',
    'Uni-Tubingen': 'University Hospital of Tuebingen',
    'Universitaetsklinikum-Essen': 'University of Essen',
    'University-Hospital-Aachen': 'University Hospital Aachen',
    'University-of-Graz': 'Medical University of Graz',
    'University-of-Kragujevac': 'University Clinical Centre Kragujevac',
    'University-of-Linz': 'Kepler University Hospital Linz',
}

# Hospital to country flag mapping (using normalized data format as keys)
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
    'New-Somerset-Hospital': '🇿🇦',
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


def get_hospital_display_name(normalized_name: str) -> str:
    """Get formal hospital name (handles trailing dashes like FINAL_fig4)."""
    # Handle trailing dashes
    if normalized_name.endswith('-'):
        normalized_name = normalized_name[:-1]

    # Return formal name if found, otherwise return with dashes replaced
    return FORMAL_HOSPITAL_NAMES.get(normalized_name, normalized_name.replace('-', ' '))


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


def process_method_results(predictions_dict, metadata_map, clinical_annotations,
                          special_mappings, mappings, hospital_weights, method_name):
    """
    Process predictions and calculate per-simulation hospital median errors.

    Returns:
        - aggregated_df: DataFrame with mean ± CI statistics for all three error types
        - raw_errors: Dict[hospital -> list of per-simulation median combined errors]
    """
    print(f"\n Processing {method_name} results...")

    # Track all three error types separately
    hospital_tip_errors = {}
    hospital_carina_errors = {}
    hospital_combined_errors = {}

    for hospital, sim_predictions_list in predictions_dict.items():
        hospital_tip_errors[hospital] = []
        hospital_carina_errors[hospital] = []
        hospital_combined_errors[hospital] = []

        for sim_idx, predictions in enumerate(sim_predictions_list):
            # Calculate errors for all images in this simulation
            error_df = calculate_errors_for_predictions(
                predictions, metadata_map, clinical_annotations,
                special_mappings, mappings, hospital_weights
            )

            # Get median for each error type for this simulation
            if len(error_df) > 0:
                # Tip error
                valid_tip = error_df['tip_error_mm'].dropna()
                if len(valid_tip) > 0:
                    hospital_tip_errors[hospital].append(valid_tip.median())

                # Carina error
                valid_carina = error_df['carina_error_mm'].dropna()
                if len(valid_carina) > 0:
                    hospital_carina_errors[hospital].append(valid_carina.median())

                # Combined error
                valid_combined = error_df['combined_tip_carina_error_mm'].dropna()
                if len(valid_combined) > 0:
                    hospital_combined_errors[hospital].append(valid_combined.median())
                else:
                    print(f"     {hospital} sim_{sim_idx}: No valid combined errors")
            else:
                print(f"     {hospital} sim_{sim_idx}: No predictions found")

    # Aggregate each error type separately
    tip_aggregated = aggregate_simulation_stats(hospital_tip_errors, use_median=True)
    carina_aggregated = aggregate_simulation_stats(hospital_carina_errors, use_median=True)
    combined_aggregated = aggregate_simulation_stats(hospital_combined_errors, use_median=True)

    # Rename columns to distinguish error types and calculate CI bounds
    tip_aggregated = tip_aggregated.rename(columns={
        'combined_error_mean_mm': 'tip_error_mean_mm',
        'combined_error_std_mm': 'tip_error_std_mm',
        'combined_error_ci_margin_mm': 'tip_error_ci_margin_mm'
    })
    tip_aggregated['tip_error_ci_lower_mm'] = tip_aggregated['tip_error_mean_mm'] - tip_aggregated['tip_error_ci_margin_mm']
    tip_aggregated['tip_error_ci_upper_mm'] = tip_aggregated['tip_error_mean_mm'] + tip_aggregated['tip_error_ci_margin_mm']

    carina_aggregated = carina_aggregated.rename(columns={
        'combined_error_mean_mm': 'carina_error_mean_mm',
        'combined_error_std_mm': 'carina_error_std_mm',
        'combined_error_ci_margin_mm': 'carina_error_ci_margin_mm'
    })
    carina_aggregated['carina_error_ci_lower_mm'] = carina_aggregated['carina_error_mean_mm'] - carina_aggregated['carina_error_ci_margin_mm']
    carina_aggregated['carina_error_ci_upper_mm'] = carina_aggregated['carina_error_mean_mm'] + carina_aggregated['carina_error_ci_margin_mm']

    # Calculate CI bounds for combined error
    combined_aggregated['combined_error_ci_lower_mm'] = combined_aggregated['combined_error_mean_mm'] - combined_aggregated['combined_error_ci_margin_mm']
    combined_aggregated['combined_error_ci_upper_mm'] = combined_aggregated['combined_error_mean_mm'] + combined_aggregated['combined_error_ci_margin_mm']

    # Merge all three error types into one DataFrame
    aggregated_df = combined_aggregated.merge(
        tip_aggregated[['hospital_name', 'tip_error_mean_mm', 'tip_error_std_mm',
                       'tip_error_ci_margin_mm', 'tip_error_ci_lower_mm', 'tip_error_ci_upper_mm']],
        on='hospital_name', how='left'
    ).merge(
        carina_aggregated[['hospital_name', 'carina_error_mean_mm', 'carina_error_std_mm',
                          'carina_error_ci_margin_mm', 'carina_error_ci_lower_mm', 'carina_error_ci_upper_mm']],
        on='hospital_name', how='left'
    )

    print(f"    Processed {len(aggregated_df)} hospitals")
    print(f"    Per-simulation errors: {sum(len(v) for v in hospital_combined_errors.values())} total")

    return aggregated_df, hospital_combined_errors


def load_direct_predictions_simple(mappings_df):
    """Load direct predictions with hospital mapping."""
    from prediction_loading_utils import load_direct_predictions

    # Just pass the DataFrame directly - load_direct_predictions handles it
    return load_direct_predictions(mappings=mappings_df)


def process_direct_results_simple(metadata_map, clinical_annotations, special_mappings,
                                 mappings, hospital_weights):
    """Process direct predictions and calculate hospital median errors."""
    from prediction_loading_utils import calculate_errors_for_predictions

    print("\n Processing Direct Prediction results...")

    # Load direct predictions
    direct_predictions = load_direct_predictions_simple(mappings)

    # Calculate errors
    error_df = calculate_errors_for_predictions(
        direct_predictions, metadata_map, clinical_annotations,
        special_mappings, mappings, hospital_weights
    )

    # Aggregate by hospital (median for all three error types)
    hospital_stats = error_df.groupby('hospital_name').agg({
        'tip_error_mm': 'median',
        'carina_error_mm': 'median',
        'combined_tip_carina_error_mm': 'median'
    }).reset_index()

    hospital_stats = hospital_stats.rename(columns={
        'tip_error_mm': 'tip_error_median_mm',
        'carina_error_mm': 'carina_error_median_mm',
        'combined_tip_carina_error_mm': 'combined_error_median_mm'
    })

    print(f"    Processed {len(hospital_stats)} hospitals")

    return hospital_stats


def perform_statistical_tests_simple(cl_raw_errors, ft_raw_errors, direct_median, hospital):
    """
    Perform statistical tests for one hospital.

    Args:
        cl_raw_errors: List of per-simulation median errors for CL
        ft_raw_errors: List of per-simulation median errors for FT
        direct_median: Single value for direct prediction
        hospital: Hospital name (for logging)

    Returns:
        dict with test results
    """
    results = {
        'cl_vs_ft_p': np.nan,
        'cl_vs_direct_p': np.nan,
        'cl_better_than_ft': '',
        'cl_better_than_direct': ''
    }

    # Remove NaN values
    cl_clean = [e for e in cl_raw_errors if not pd.isna(e)]
    ft_clean = [e for e in ft_raw_errors if not pd.isna(e)]

    if len(cl_clean) == 0 or len(ft_clean) == 0:
        return results

    # Two-sample t-test: CL vs FT
    if len(cl_clean) >= 2 and len(ft_clean) >= 2:
        try:
            t_stat, p_value = scipy_stats.ttest_ind(cl_clean, ft_clean)
            results['cl_vs_ft_p'] = p_value

            cl_mean = np.mean(cl_clean)
            ft_mean = np.mean(ft_clean)

            if p_value < 0.05:
                if cl_mean < ft_mean:
                    results['cl_better_than_ft'] = 'CL < FT*'
                else:
                    results['cl_better_than_ft'] = 'CL > FT*'
            else:
                results['cl_better_than_ft'] = 'NS'
        except:
            pass

    # One-sample t-test: CL vs Direct
    if len(cl_clean) >= 2 and not pd.isna(direct_median):
        try:
            t_stat, p_value = scipy_stats.ttest_1samp(cl_clean, direct_median)
            results['cl_vs_direct_p'] = p_value

            cl_mean = np.mean(cl_clean)

            if p_value < 0.05:
                if cl_mean < direct_median:
                    results['cl_better_than_direct'] = 'CL < Direct*'
                else:
                    results['cl_better_than_direct'] = 'CL > Direct*'
            else:
                results['cl_better_than_direct'] = 'NS'
        except:
            pass

    return results


def create_two_method_comparison_plot(cl_df, ft_df, cl_raw_errors, ft_raw_errors, output_dir):
    """Create horizontal dot plot with error bars comparing CL vs FT."""

    print("\n Creating CL vs FT comparison plot...")

    # Filter out New Somerset Hospital
    cl_df = cl_df[~cl_df['hospital_name'].str.contains('Somerset', case=False, na=False)]
    ft_df = ft_df[~ft_df['hospital_name'].str.contains('Somerset', case=False, na=False)]

    common_hospitals = list(set(cl_df['hospital_name']) & set(ft_df['hospital_name']))

    if not common_hospitals:
        print(" No common hospitals found!")
        return

    print(f"   Plotting {len(common_hospitals)} hospitals")

    # Sort by FT mean error (highest to lowest)
    hospital_order = []
    for hospital in common_hospitals:
        ft_mean = ft_df[ft_df['hospital_name'] == hospital]['combined_error_mean_mm'].iloc[0]
        hospital_order.append((hospital, ft_mean))

    hospital_order.sort(key=lambda x: x[1], reverse=True)
    ordered_hospitals = [h[0] for h in hospital_order]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 14))

    # Colors
    cl_color = '#DC143C'  # Crimson
    ft_color = '#0066CC'  # Blue

    # Plot dots with error bars
    y_positions = np.arange(len(ordered_hospitals))

    for i, hospital in enumerate(ordered_hospitals):
        cl_row = cl_df[cl_df['hospital_name'] == hospital].iloc[0]
        ft_row = ft_df[ft_df['hospital_name'] == hospital].iloc[0]

        cl_mean = cl_row['combined_error_mean_mm'] / 10.0  # Convert to cm
        ft_mean = ft_row['combined_error_mean_mm'] / 10.0

        cl_ci_margin = cl_row['combined_error_ci_margin_mm'] / 10.0
        ft_ci_margin = ft_row['combined_error_ci_margin_mm'] / 10.0

        # Plot error bars and dots
        ax.errorbar(ft_mean, i, xerr=ft_ci_margin, fmt='o', color=ft_color,
                   markersize=8, alpha=0.8, capsize=3, capthick=1.5)
        ax.errorbar(cl_mean, i, xerr=cl_ci_margin, fmt='o', color=cl_color,
                   markersize=8, alpha=0.8, capsize=3, capthick=1.5)

    # Calculate averages
    avg_cl = cl_df[cl_df['hospital_name'].isin(common_hospitals)]['combined_error_mean_mm'].mean() / 10.0
    avg_ft = ft_df[ft_df['hospital_name'].isin(common_hospitals)]['combined_error_mean_mm'].mean() / 10.0

    # Add vertical lines for averages
    ax.axvline(x=avg_cl, color=cl_color, linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axvline(x=avg_ft, color=ft_color, linestyle=':', linewidth=1.5, alpha=0.7)

    # Set y-axis labels with space for flags
    hospital_labels = [get_hospital_display_name(h) + '          ' for h in ordered_hospitals]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(hospital_labels, fontsize=10)

    # Add flags
    for i, hospital in enumerate(ordered_hospitals):
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
            ab = AnnotationBbox(imagebox, (-0.02, i),
                              xycoords=trans, frameon=False,
                              box_alignment=(1, 0.5), pad=0)
            ax.add_artist(ab)
        except:
            pass

    # Set x-axis label
    ax.set_xlabel('Combined ETT and Carina Localization Error (cm)', fontsize=12, color='black')

    # Set x-axis limits
    all_errors = []
    for hospital in ordered_hospitals:
        cl_row = cl_df[cl_df['hospital_name'] == hospital].iloc[0]
        ft_row = ft_df[ft_df['hospital_name'] == hospital].iloc[0]
        cl_mean = cl_row['combined_error_mean_mm'] / 10.0
        ft_mean = ft_row['combined_error_mean_mm'] / 10.0
        cl_ci = cl_row['combined_error_ci_margin_mm'] / 10.0
        ft_ci = ft_row['combined_error_ci_margin_mm'] / 10.0
        all_errors.extend([cl_mean - cl_ci, cl_mean + cl_ci, ft_mean - ft_ci, ft_mean + ft_ci])

    min_error = min(all_errors)
    max_error = max(all_errors)
    padding = (max_error - min_error) * 0.1
    ax.set_xlim(min_error - padding, max_error + padding)

    # Remove grid
    ax.grid(False)

    # Make spines black
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

    # Add legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=ft_color,
               markersize=8, alpha=0.8, label='Fine-tuning'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=cl_color,
               markersize=8, alpha=0.8, label='Continual Learning')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True,
             fancybox=False, edgecolor='black', fontsize=10)

    # Add text annotations for averages
    y_top = ax.get_ylim()[1]
    trans = ax.transData
    ax.text(avg_ft, y_top + 0.15,
           f'Fine-tuning\nAvg: {avg_ft:.2f} cm',
           ha='center', va='bottom', fontsize=9, color=ft_color,
           transform=trans, fontweight='bold')
    ax.text(avg_cl, y_top + 0.15,
           f'Continual Learning\nAvg: {avg_cl:.2f} cm',
           ha='center', va='bottom', fontsize=9, color=cl_color,
           transform=trans, fontweight='bold')

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(left=0.25)

    # Save figure
    output_path = output_dir / 'etable4_cl_vs_ft_comparison_ci95.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f" Saved plot: {output_path}")
    print(f"   CL Average: {avg_cl:.2f} cm")
    print(f"   FT Average: {avg_ft:.2f} cm")


def create_three_method_comparison_plot(cl_df, ft_df, direct_df, cl_raw_errors, ft_raw_errors, output_dir):
    """Create horizontal dot plot with error bars comparing CL vs FT vs Direct."""

    print("\n Creating CL vs FT vs Direct comparison plot...")

    # Filter out New Somerset Hospital
    cl_df = cl_df[~cl_df['hospital_name'].str.contains('Somerset', case=False, na=False)]
    ft_df = ft_df[~ft_df['hospital_name'].str.contains('Somerset', case=False, na=False)]
    direct_df = direct_df[~direct_df['hospital_name'].str.contains('Somerset', case=False, na=False)]

    common_hospitals = list(set(cl_df['hospital_name']) & set(ft_df['hospital_name']) & set(direct_df['hospital_name']))

    if not common_hospitals:
        print(" No common hospitals found!")
        return

    print(f"   Plotting {len(common_hospitals)} hospitals")

    # Sort by direct error (highest to lowest)
    hospital_order = []
    for hospital in common_hospitals:
        direct_error = direct_df[direct_df['hospital_name'] == hospital]['combined_error_median_mm'].iloc[0]
        hospital_order.append((hospital, direct_error))

    hospital_order.sort(key=lambda x: x[1], reverse=True)
    ordered_hospitals = [h[0] for h in hospital_order]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 14))

    # Colors
    cl_color = '#DC143C'  # Crimson
    ft_color = '#0066CC'  # Blue
    direct_color = '#228B22'  # Forest green

    # Plot dots with error bars
    y_positions = np.arange(len(ordered_hospitals))

    for i, hospital in enumerate(ordered_hospitals):
        cl_row = cl_df[cl_df['hospital_name'] == hospital].iloc[0]
        ft_row = ft_df[ft_df['hospital_name'] == hospital].iloc[0]
        direct_row = direct_df[direct_df['hospital_name'] == hospital].iloc[0]

        cl_mean = cl_row['combined_error_mean_mm'] / 10.0
        ft_mean = ft_row['combined_error_mean_mm'] / 10.0
        direct_mean = direct_row['combined_error_median_mm'] / 10.0

        cl_ci_margin = cl_row['combined_error_ci_margin_mm'] / 10.0
        ft_ci_margin = ft_row['combined_error_ci_margin_mm'] / 10.0

        # Plot all three methods
        ax.scatter(direct_mean, i, color=direct_color, s=100, alpha=0.8, zorder=3, marker='o')
        ax.errorbar(ft_mean, i, xerr=ft_ci_margin, fmt='o', color=ft_color,
                   markersize=8, alpha=0.8, capsize=3, capthick=1.5)
        ax.errorbar(cl_mean, i, xerr=cl_ci_margin, fmt='o', color=cl_color,
                   markersize=8, alpha=0.8, capsize=3, capthick=1.5)

    # Calculate averages
    avg_cl = cl_df[cl_df['hospital_name'].isin(common_hospitals)]['combined_error_mean_mm'].mean() / 10.0
    avg_ft = ft_df[ft_df['hospital_name'].isin(common_hospitals)]['combined_error_mean_mm'].mean() / 10.0
    avg_direct = direct_df[direct_df['hospital_name'].isin(common_hospitals)]['combined_error_median_mm'].mean() / 10.0

    # Add vertical lines for averages
    ax.axvline(x=avg_cl, color=cl_color, linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axvline(x=avg_ft, color=ft_color, linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axvline(x=avg_direct, color=direct_color, linestyle=':', linewidth=1.5, alpha=0.7)

    # Set y-axis labels with space for flags
    hospital_labels = [get_hospital_display_name(h) + '          ' for h in ordered_hospitals]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(hospital_labels, fontsize=10)

    # Add flags
    for i, hospital in enumerate(ordered_hospitals):
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
            ab = AnnotationBbox(imagebox, (-0.02, i),
                              xycoords=trans, frameon=False,
                              box_alignment=(1, 0.5), pad=0)
            ax.add_artist(ab)
        except:
            pass

    # Set x-axis label
    ax.set_xlabel('Combined ETT and Carina Localization Error (cm)', fontsize=12, color='black')

    # Set x-axis limits
    all_errors = []
    for hospital in ordered_hospitals:
        cl_row = cl_df[cl_df['hospital_name'] == hospital].iloc[0]
        ft_row = ft_df[ft_df['hospital_name'] == hospital].iloc[0]
        direct_row = direct_df[direct_df['hospital_name'] == hospital].iloc[0]
        cl_mean = cl_row['combined_error_mean_mm'] / 10.0
        ft_mean = ft_row['combined_error_mean_mm'] / 10.0
        direct_mean = direct_row['combined_error_median_mm'] / 10.0
        cl_ci = cl_row['combined_error_ci_margin_mm'] / 10.0
        ft_ci = ft_row['combined_error_ci_margin_mm'] / 10.0
        all_errors.extend([cl_mean - cl_ci, cl_mean + cl_ci, ft_mean - ft_ci, ft_mean + ft_ci, direct_mean])

    min_error = min(all_errors)
    max_error = max(all_errors)
    padding = (max_error - min_error) * 0.1
    ax.set_xlim(min_error - padding, max_error + padding)

    # Remove grid
    ax.grid(False)

    # Make spines black
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=cl_color, alpha=0.8, label='Continual Learning (CL)'),
        Patch(facecolor=ft_color, alpha=0.8, label='Fine-tuning (FT)'),
        Patch(facecolor=direct_color, alpha=0.8, label='Direct Prediction (Direct)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True,
             fancybox=False, edgecolor='black', fontsize=10)

    # Add text annotations for averages with offsets to avoid overlap
    y_top = ax.get_ylim()[1]
    trans = ax.transData
    ax.text(avg_cl - 0.3, y_top + 0.15,
           f'CL Avg:\n{avg_cl:.2f} cm',
           ha='center', va='bottom', fontsize=9, color=cl_color,
           transform=trans, fontweight='bold')
    ax.text(avg_ft + 0.3, y_top + 0.15,
           f'FT Avg:\n{avg_ft:.2f} cm',
           ha='center', va='bottom', fontsize=9, color=ft_color,
           transform=trans, fontweight='bold')
    ax.text(avg_direct, y_top + 0.15,
           f'Direct Avg:\n{avg_direct:.2f} cm',
           ha='center', va='bottom', fontsize=9, color=direct_color,
           transform=trans, fontweight='bold')

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(left=0.25)

    # Save figure
    output_path = output_dir / 'etable4_three_method_comparison_ci95.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f" Saved plot: {output_path}")
    print(f"   CL Average: {avg_cl:.2f} cm")
    print(f"   FT Average: {avg_ft:.2f} cm")
    print(f"   Direct Average: {avg_direct:.2f} cm")


def create_etable4(cl_df, ft_df, direct_df, cl_raw_errors, ft_raw_errors, output_dir):
    """Create eTable 4 with statistical comparisons."""

    print("\n Creating eTable 4...")

    # Filter out New Somerset Hospital
    cl_df = cl_df[~cl_df['hospital_name'].str.contains('Somerset', case=False, na=False)]
    ft_df = ft_df[~ft_df['hospital_name'].str.contains('Somerset', case=False, na=False)]
    direct_df = direct_df[~direct_df['hospital_name'].str.contains('Somerset', case=False, na=False)]

    common_hospitals = list(set(cl_df['hospital_name']) & set(ft_df['hospital_name']) & set(direct_df['hospital_name']))

    # Sort alphabetically by display name
    common_hospitals = sorted(common_hospitals, key=lambda h: get_hospital_display_name(h))

    print(f"   Creating table for {len(common_hospitals)} hospitals")

    # Prepare table data
    table_data = []
    table_data.append([
        'Hospital', 'Method',
        'ETT Tip Error (cm)', 'Carina Error (cm)', 'Combined Error (cm)',
        'CL vs FT p-value', 'CL vs Direct p-value', 'CL vs FT', 'CL vs Direct'
    ])

    for hospital in common_hospitals:
        display_name = get_hospital_display_name(hospital)

        # Get data for this hospital
        cl_row = cl_df[cl_df['hospital_name'] == hospital].iloc[0]
        ft_row = ft_df[ft_df['hospital_name'] == hospital].iloc[0]
        direct_row = direct_df[direct_df['hospital_name'] == hospital].iloc[0]

        # Perform statistical tests
        cl_errors = cl_raw_errors.get(hospital, [])
        ft_errors = ft_raw_errors.get(hospital, [])
        direct_median = direct_row['combined_error_median_mm']

        stats = perform_statistical_tests_simple(cl_errors, ft_errors, direct_median, hospital)

        # Format p-values
        cl_vs_ft_p_str = f"{stats['cl_vs_ft_p']:.3f}" if not pd.isna(stats['cl_vs_ft_p']) else ''
        if not pd.isna(stats['cl_vs_ft_p']) and stats['cl_vs_ft_p'] < 0.001:
            cl_vs_ft_p_str = '<0.001'

        cl_vs_direct_p_str = f"{stats['cl_vs_direct_p']:.3f}" if not pd.isna(stats['cl_vs_direct_p']) else ''
        if not pd.isna(stats['cl_vs_direct_p']) and stats['cl_vs_direct_p'] < 0.001:
            cl_vs_direct_p_str = '<0.001'

        # Add CL row
        cl_tip = cl_row['tip_error_mean_mm'] / 10.0
        cl_carina = cl_row['carina_error_mean_mm'] / 10.0
        cl_combined = cl_row['combined_error_mean_mm'] / 10.0
        cl_tip_ci_lower = cl_row['tip_error_ci_lower_mm'] / 10.0
        cl_tip_ci_upper = cl_row['tip_error_ci_upper_mm'] / 10.0
        cl_carina_ci_lower = cl_row['carina_error_ci_lower_mm'] / 10.0
        cl_carina_ci_upper = cl_row['carina_error_ci_upper_mm'] / 10.0
        cl_combined_ci_lower = cl_row['combined_error_ci_lower_mm'] / 10.0
        cl_combined_ci_upper = cl_row['combined_error_ci_upper_mm'] / 10.0

        table_data.append([
            display_name,
            'Continual Learning',
            f"{cl_tip:.2f} ({cl_tip_ci_lower:.2f}-{cl_tip_ci_upper:.2f})",
            f"{cl_carina:.2f} ({cl_carina_ci_lower:.2f}-{cl_carina_ci_upper:.2f})",
            f"{cl_combined:.2f} ({cl_combined_ci_lower:.2f}-{cl_combined_ci_upper:.2f})",
            cl_vs_ft_p_str,
            cl_vs_direct_p_str,
            stats['cl_better_than_ft'],
            stats['cl_better_than_direct']
        ])

        # Add FT row
        ft_tip = ft_row['tip_error_mean_mm'] / 10.0
        ft_carina = ft_row['carina_error_mean_mm'] / 10.0
        ft_combined = ft_row['combined_error_mean_mm'] / 10.0
        ft_tip_ci_lower = ft_row['tip_error_ci_lower_mm'] / 10.0
        ft_tip_ci_upper = ft_row['tip_error_ci_upper_mm'] / 10.0
        ft_carina_ci_lower = ft_row['carina_error_ci_lower_mm'] / 10.0
        ft_carina_ci_upper = ft_row['carina_error_ci_upper_mm'] / 10.0
        ft_combined_ci_lower = ft_row['combined_error_ci_lower_mm'] / 10.0
        ft_combined_ci_upper = ft_row['combined_error_ci_upper_mm'] / 10.0

        table_data.append([
            '',
            'Fine-tuning',
            f"{ft_tip:.2f} ({ft_tip_ci_lower:.2f}-{ft_tip_ci_upper:.2f})",
            f"{ft_carina:.2f} ({ft_carina_ci_lower:.2f}-{ft_carina_ci_upper:.2f})",
            f"{ft_combined:.2f} ({ft_combined_ci_lower:.2f}-{ft_combined_ci_upper:.2f})",
            '', '', '', ''
        ])

        # Add Direct row
        direct_tip = direct_row['tip_error_median_mm'] / 10.0
        direct_carina = direct_row['carina_error_median_mm'] / 10.0
        direct_combined = direct_row['combined_error_median_mm'] / 10.0

        table_data.append([
            '',
            'Direct Prediction',
            f"{direct_tip:.2f}",
            f"{direct_carina:.2f}",
            f"{direct_combined:.2f}",
            '', '', '', ''
        ])

    # Create DataFrame
    df = pd.DataFrame(table_data[1:], columns=table_data[0])

    # Save to CSV
    csv_path = output_dir / 'eTable4_localization_errors_with_stats.csv'
    df.to_csv(csv_path, index=False)
    print(f" Saved table (CSV): {csv_path}")

    # Save text version
    txt_path = output_dir / 'eTable4_localization_errors_with_stats.txt'
    with open(txt_path, 'w') as f:
        f.write("eTable 4. Localization Error Comparison Across Methods with Statistical Analysis\n")
        f.write("="*120 + "\n\n")
        f.write("Values shown as: Mean (95% CI Lower-Upper) for CL and FT, Median for Direct Prediction\n")
        f.write("Statistical tests: Two-sample t-test for CL vs FT, One-sample t-test for CL vs Direct\n")
        f.write("* indicates p < 0.05, NS = not significant\n\n")
        f.write(df.to_string(index=False))

    print(f" Saved table (TXT): {txt_path}")


def main():
    """Main function to create eTable 4."""

    parser = argparse.ArgumentParser(description="Create eTable 4 with figures")
    parser.add_argument("--sanity-check", action="store_true",
                       help="Only process first simulation for quick testing")
    args = parser.parse_args()

    num_sims = 1 if args.sanity_check else 10

    print("="*80)
    print("CREATING eTABLE 4: LOCALIZATION ERROR COMPARISON")
    print("Using standardized utility functions")
    if args.sanity_check:
        print("  SANITY CHECK MODE: Processing only 1 simulation")
    print("="*80)

    # Use paths from configuration
    output_dir = OUTPUT_DIR
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load data following test_prediction_loading_utils.py pattern
    print("\n Loading common dependencies...")
    mappings = pd.read_csv(MAPPINGS_PATH)
    hospital_weights = load_hospital_weights_standardized(DEMOGRAPHICS_DIR)
    clinical_annotations, special_mappings = load_all_clinical_annotations()
    metadata_map = load_annotation_metadata(ANNOTATIONS_PATH)

    print(f"    Loaded {len(mappings)} ID mappings")
    print(f"    Loaded weight data for {len(hospital_weights)} hospitals")
    print(f"    Loaded {len(clinical_annotations)} clinical annotations")
    print(f"    Loaded metadata for {len(metadata_map)} images")

    # Load raw predictions (LEVEL 1)
    print("\n" + "="*80)
    print("LEVEL 1: LOADING RAW PREDICTIONS")
    print("="*80)

    cl_predictions = load_cl_predictions_multi_sim(num_simulations=num_sims)
    print(f"    Loaded CL predictions for {len(cl_predictions)} hospitals")

    ft_predictions = load_ft_predictions_multi_sim(num_simulations=num_sims)
    print(f"    Loaded FT predictions for {len(ft_predictions)} hospitals")

    # Process and calculate errors (LEVEL 2)
    print("\n" + "="*80)
    print("LEVEL 2: CALCULATING ERRORS AND STATISTICS")
    print("="*80)

    cl_df, cl_raw_errors = process_method_results(
        cl_predictions, metadata_map, clinical_annotations,
        special_mappings, mappings, hospital_weights, "CL"
    )

    ft_df, ft_raw_errors = process_method_results(
        ft_predictions, metadata_map, clinical_annotations,
        special_mappings, mappings, hospital_weights, "FT"
    )

    direct_df = process_direct_results_simple(
        metadata_map, clinical_annotations, special_mappings,
        mappings, hospital_weights
    )

    # Validation checks
    print("\n" + "="*80)
    print("VALIDATION CHECKS")
    print("="*80)

    print(f"\n Hospital counts:")
    print(f"  - CL: {len(cl_df)} hospitals")
    print(f"  - FT: {len(ft_df)} hospitals")
    print(f"  - Direct: {len(direct_df)} hospitals")

    # Check for NaN values in raw errors
    nan_warnings = False
    for hospital, errors in cl_raw_errors.items():
        nan_count = sum(1 for e in errors if pd.isna(e))
        if nan_count > 0:
            print(f"    CL {hospital}: {nan_count}/{len(errors)} simulations have NaN values")
            nan_warnings = True

    for hospital, errors in ft_raw_errors.items():
        nan_count = sum(1 for e in errors if pd.isna(e))
        if nan_count > 0:
            print(f"    FT {hospital}: {nan_count}/{len(errors)} simulations have NaN values")
            nan_warnings = True

    if not nan_warnings:
        print("   No NaN values detected in any simulation results")

    # Generate figures and tables
    print("\n" + "="*80)
    print("LEVEL 3: GENERATING FIGURES AND TABLES")
    print("="*80)

    # Create two-method comparison plot (CL vs FT)
    create_two_method_comparison_plot(cl_df, ft_df, cl_raw_errors, ft_raw_errors, output_dir)

    # Create three-method comparison plot (CL vs FT vs Direct)
    create_three_method_comparison_plot(cl_df, ft_df, direct_df, cl_raw_errors, ft_raw_errors, output_dir)

    # Create eTable 4
    create_etable4(cl_df, ft_df, direct_df, cl_raw_errors, ft_raw_errors, output_dir)

    print("\n" + "="*80)
    print(" ALL TASKS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print(f"  1. {output_dir}/etable4_cl_vs_ft_comparison_ci95.png")
    print(f"  2. {output_dir}/etable4_three_method_comparison_ci95.png")
    print(f"  3. {output_dir}/eTable4_localization_errors_with_stats.csv")
    print(f"  4. {output_dir}/eTable4_localization_errors_with_stats.txt")
    if args.sanity_check:
        print(f"\n  Note: Ran in sanity check mode with only {num_sims} simulation(s)")
        print(f"   For final results, run without --sanity-check to process all 10 simulations")


if __name__ == "__main__":
    main()
