#!/usr/bin/env python3
"""
Holdout Analysis for Continual Learning - Version 3
This version directly uses the trained model without saving/loading
Designed for use with auto-scheduling submission script
"""

import argparse
import os
import sys
import yaml
import pandas as pd
import time
from datetime import datetime
from tqdm import tqdm
import json
import shutil

# Note: Run from the run_CL directory - uses relative imports

# Ensure wandb is disabled for holdout experiments
os.environ['WANDB_MODE'] = 'disabled'

# Import the main continual learning function
from continual_learning import process_hospital_sequentially
from utils.constants import *

# Override WANDB_OFF to disable wandb
import utils.constants as constants
constants.WANDB_OFF = True

# Import wandb and initialize in offline mode to prevent None errors
import wandb
wandb.init(mode="disabled", project="holdout-test")

from utils.common_helpers import get_hospital_data_annos_loaders, get_model, format_results
from utils.fine_tune_helpers import calculate_loss_and_error
from utils.utils import normalize_hospital_name
from utils.config_helpers import get_output_path_for_global_CL

def run_holdout_experiment(config, holdout_hospital, test_mode=False):
    """
    Run continual learning excluding holdout, then test on holdout
    """

    print(f"\n Starting holdout experiment for: {holdout_hospital}")

    # Get all hospital data loaders
    all_data_loaders = get_hospital_data_annos_loaders(config)

    # Get list of hospitals
    all_hospitals = list(all_data_loaders.keys())
    all_hospitals = [h for h in all_hospitals if h != ALL_KEY]

    # In test mode, limit to just our 3 test hospitals
    if test_mode:
        test_hospitals = ["American-University-of-Beirut", "Childrens-Hospital-Colorado", "Indus"]
        all_hospitals = [h for h in all_hospitals if h in test_hospitals]
        print(f"   🧪 TEST MODE: Limited to {len(all_hospitals)} hospitals: {all_hospitals}")

    # Remove holdout from training list
    training_hospitals = [h for h in all_hospitals if h != holdout_hospital]

    print(f"    Training on {len(training_hospitals)} hospitals")
    print(f"    Holding out: {holdout_hospital}")

    # Filter data loaders to only include training hospitals
    training_data_loaders = {h: all_data_loaders[h] for h in training_hospitals}
    # Keep ALL_KEY for research evaluation if it exists
    if ALL_KEY in all_data_loaders:
        training_data_loaders[ALL_KEY] = all_data_loaders[ALL_KEY]

    # Get the model
    use_random_init = config.get("use_random_init", False)
    model_path = config.get("model_path", "")
    model = get_model(config, model_path, config[UPDATE_METHOD], use_random_init=use_random_init)

    # Create output directory if it doesn't exist
    output_path = config['output_path']
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
        print(f"    Created output directory: {output_path}")

    # Get simulation index from config (set by submission script via skip_simulation)
    simulation_idx = config.get('skip_simulation', 0)

    # Randomize hospital order based on simulation index for reproducibility
    import random
    random.seed(simulation_idx)
    random.shuffle(training_hospitals)

    print(f"\n Training Phase - Processing hospitals sequentially (simulation {simulation_idx})...")
    print(f"    Hospital order for this simulation: {training_hospitals}")

    # Save hospital order to file
    hospitals_order_str = ",".join(training_hospitals)
    order_file = os.path.join(output_path, f"hospitals_order_sim{simulation_idx}.txt")
    with open(order_file, "w") as f:
        f.write(hospitals_order_str)
    print(f"    Hospital order saved to: {order_file}")

    # Set up update dict (similar to continual_learning.py)
    update_dict = {
        ITERATION_IDX_INIT: 0,
        PATIENT_IDX_INIT: -1,
        UPDATE_METHOD: config[UPDATE_METHOD],
        OUTPUT_PATH: output_path,
        SIMULATION_IDX: simulation_idx,
        HAS_L2INIT: config.get(HAS_L2INIT, False),
        'holdout_hospital': holdout_hospital,  # For unique temp_models directories in parallel runs
    }

    # Add parameters for fixed epochs if specified
    if 'num_epochs_per_hospital' in config and config['num_epochs_per_hospital'] is not None:
        update_dict['num_epochs_per_hospital'] = config['num_epochs_per_hospital']
    if 'num_iterations' in config:
        update_dict['num_iterations'] = config['num_iterations']
    if 'cv_folds' in config:
        update_dict['cv_folds'] = config['cv_folds']

    # Perform continual learning on training hospitals
    training_results, skipped_hospitals = process_hospital_sequentially(
        model=model,
        hospitals_order=training_hospitals,
        data_annos_loaders=training_data_loaders,
        update_dict=update_dict,
        config=config
    )

    # Check if any hospitals failed - if so, abort this simulation
    if skipped_hospitals:
        print(f"\n SIMULATION FAILED: {len(skipped_hospitals)} hospitals had errors during training")
        print(f"   Cleaning up simulation {simulation_idx} files...")

        # Delete only the hospital order file for this simulation (simulation_dir doesn't exist yet)
        order_file = os.path.join(output_path, f"hospitals_order_sim{simulation_idx}.txt")
        if os.path.exists(order_file):
            os.remove(order_file)
            print(f"     Deleted: {order_file}")

        print(f"   This simulation will be retried later by the scheduler")
        sys.exit(1)  # Exit with error code so bash script knows it failed

    print(f"\n Training completed on {len(training_hospitals)} hospitals")

    # Now test on holdout hospital using the trained model
    print(f"\n🧪 Testing on holdout hospital: {holdout_hospital}")

    # Get holdout hospital's test data
    holdout_test_loader = all_data_loaders[holdout_hospital][TEST_DATA_SOURCE]

    # Perform inference only (no training) on holdout
    holdout_predictions = []
    patient_idx = -1

    # Use the same inference loop as in CL_helpers.py
    for batch_idx, batch in tqdm(enumerate(holdout_test_loader[DATA_LOADERS_KEY])):
        images, image_ids = batch["image"], batch["image_id"].tolist()
        batch_size = len(images)

        # Predict using the trained model
        predictions = model.predict(zip(images, image_ids))

        # Calculate metrics
        mean_loss_err = calculate_loss_and_error(predictions, holdout_test_loader[ANNOS_LOADER_KEY])

        print(f'   Carina error: {mean_loss_err[f"{ANNO_CAT_CARINA}{ERROR_SUFFIX}"]:.2f} | Tip error: {mean_loss_err[f"{ANNO_CAT_TIP}{ERROR_SUFFIX}"]:.2f}')

        # Format results (same as CL does)
        holdout_update_dict = {
            PATIENT_IDX_INIT: patient_idx,
            BATCH_IDX: batch_idx,
            BATCH_SIZE: batch_size,
            ITERATION_IDX_INIT: -1,  # No training iteration for holdout
            SIMULATION_IDX: simulation_idx,
        }

        holdout_predictions += format_results(
            predictions,
            holdout_test_loader[ANNOS_LOADER_KEY],
            holdout_update_dict,
        )
        patient_idx += batch_size

    # Convert to DataFrame
    holdout_df = pd.DataFrame(holdout_predictions)
    holdout_df[INDEX_HOSPITAL_FIELD] = holdout_hospital
    holdout_df['is_holdout'] = True

    print(f"    Generated {len(holdout_df)} predictions for holdout hospital")

    # Calculate summary statistics
    if 'error' in holdout_df.columns:
        mean_error = holdout_df['error'].mean()
        std_error = holdout_df['error'].std()
        print(f"    Holdout Mean error: {mean_error:.2f} ± {std_error:.2f}")

    return training_results, holdout_df

def main():
    parser = argparse.ArgumentParser(description='Run CL with holdout hospital')
    parser.add_argument('-c', '--config_path', required=True,
                        help='Config file path')
    parser.add_argument('--holdout-hospital', required=True,
                        help='Hospital to hold out from training')
    parser.add_argument('--random-init', action='store_true',
                        help='Use random initialization instead of pretrained weights')
    parser.add_argument('--test-mode', action='store_true',
                        help='Test mode with limited hospitals (for testing)')

    args = parser.parse_args()

    # Load config
    with open(args.config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Set random init if requested
    if args.random_init:
        config['use_random_init'] = True

    # Normalize hospital name
    holdout_hospital = normalize_hospital_name(args.holdout_hospital)

    # Get simulation index from config
    simulation_idx = config.get('skip_simulation', 0)

    print("=" * 70)
    print("HOLDOUT ANALYSIS V3 - Direct Model Usage")
    print("=" * 70)
    print(f"Holdout hospital: {holdout_hospital}")
    print(f"Simulation: {simulation_idx}")
    print(f"Config: {args.config_path}")
    print(f"Random init: {args.random_init}")
    print("=" * 70)

    # Turn off wandb if not already
    if config.get('wandb_off', False):
        os.environ['WANDB_MODE'] = 'disabled'

    try:
        # Run the experiment
        training_results, holdout_results = run_holdout_experiment(config, holdout_hospital, test_mode=args.test_mode)

        # Save results in nested structure: {hospital}_holdout/simulation_N/
        hospital_dir = config.get('output_path', 'outputs/holdout_v3')
        simulation_dir = os.path.join(hospital_dir, f"simulation_{simulation_idx}")
        os.makedirs(simulation_dir, exist_ok=True)

        print(f"\n Saving results to: {simulation_dir}")

        # Save training results for each hospital
        for i, df in enumerate(training_results):
            training_path = os.path.join(simulation_dir, f"training_hospital_{i}.csv")
            df.to_csv(training_path, index=False)
            print(f"    Saved training results for hospital {i}")

        # Save holdout test results
        holdout_path = os.path.join(simulation_dir, "holdout_test_results.csv")
        holdout_results.to_csv(holdout_path, index=False)
        print(f"    Saved holdout results: holdout_test_results.csv")

        # Save metadata
        metadata = {
            'holdout_hospital': holdout_hospital,
            'simulation_number': simulation_idx,
            'num_training_hospitals': len(training_results),
            'random_init': args.random_init,
            'num_holdout_predictions': len(holdout_results),
            'mean_error': float(holdout_results['error'].mean()) if 'error' in holdout_results.columns else None,
            'std_error': float(holdout_results['error'].std()) if 'error' in holdout_results.columns else None,
            'timestamp': datetime.now().isoformat(),
            'output_path': simulation_dir
        }

        metadata_path = os.path.join(simulation_dir, "holdout_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"    Saved metadata: holdout_metadata.json")

        print("\n Holdout experiment completed successfully!")
        print(f" All results in: {simulation_dir}")
        return 0

    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
