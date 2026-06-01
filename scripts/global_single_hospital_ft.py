#!/usr/bin/env python3
"""
Single-Hospital Fine-Tuning Experiment

For each hospital, fine-tunes a fresh CarinaNet model on that hospital's 50 training
samples and evaluates on that hospital's test set. Uses the EXACT same methodology
as continual learning -- same find_epoch() cross-validation, same training loop,
same inference pipeline. The only difference: weights are NOT transferred between
hospitals (fresh pretrained model each time).

Core functions used (all from existing CL codebase):
  - process_hospital_sequentially() [continual_learning.py] -- called with 1-element list
  - perform_continual_learning()    [utils/CL_helpers.py]    -- find_epoch -> train -> infer
  - find_epoch()                    [utils/training_helper.py]-- K-fold CV epoch selection
  - format_results()                [utils/common_helpers.py] -- standard CSV formatting
  - get_model()                     [utils/common_helpers.py] -- load pretrained CarinaNet
  - get_hospital_data_annos_loaders() [utils/common_helpers.py] -- load all hospital data

Usage:
    python global_single_hospital_ft.py -c <config> --hospital-group 1 --num-sims 10
    python global_single_hospital_ft.py -c <config> --hospital-group 2 --num-sims 10
    python global_single_hospital_ft.py -c <config> --hospital-group 3 --num-sims 10
    python global_single_hospital_ft.py -c <config> --hospital-group 1 --num-sims 1 --test-mode
"""

import os
import sys

# Make the repository root importable so `utils`, `models`, and sibling
# scripts resolve when this file is run directly (python <path>/script.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import gc
import shutil
import yaml
import pandas as pd
import torch
import numpy as np
import json
import random
from datetime import datetime

# Disable wandb
os.environ['WANDB_MODE'] = 'disabled'

from continual_learning import process_hospital_sequentially
from utils.constants import *

import utils.constants as constants
constants.WANDB_OFF = True

import wandb
wandb.init(mode="disabled", project="single-hospital-ft")

from utils.common_helpers import get_hospital_data_annos_loaders, get_model
from utils.config_helpers import get_model_path

# Hospital groups for parallel execution; pass --hospital-group {1, 2, 3} to
# fine-tune one group at a time (e.g. as separate SLURM jobs).
HOSPITAL_GROUPS = {
    1: [
        'Alberta',
        'American-University-of-Beirut',
        'Childrens-Hospital-Colorado',
        'Chulalongkorn-University',
        'Dr-Sardjito-Hospital',
        'Fundacion-Santa-Fe-de-Bogota',
        'Indus',
        'International-Islamic-Medical-University',
        'Istanbul-Training-Research',
        'King-Abdulaziz-Hospital',
        'Kirikkale-Hospital-',
    ],
    2: [
        'Sidra-Health',
        'Tel-Aviv-Medical-Center',
        'Tri-Service-General-Hospital',
        'Uni-Tubingen',
        'Universitaetsklinikum-Essen',
        'University-Hospital-Aachen',
        'University-of-Graz',
        'University-of-Kragujevac',
        'University-of-Linz',
    ],
    3: [
        'La-Paz-University-Hospital',
        'Maharaj-Nakorn-Chiang-Mai-Hospital',
        'Medical-Center-of-South',
        'National-Cheng-Kung-University-Hospital',
        'National-University-Singapore',
        'Newark-Beth-Israel',
        'Osaka-Metropolitan-University',
        'Puerta-del-Mar-University-Hosptial',
        'SES',
        'Shiraz-University',
    ],
}


def is_simulation_complete(sim_dir):
    """A simulation is complete only if the final CSV exists."""
    return os.path.exists(os.path.join(sim_dir, "training_hospital_0.csv"))


def clean_incomplete_simulation(sim_dir):
    """Remove partial artifacts from an incomplete simulation so it can be re-run."""
    if os.path.exists(sim_dir):
        for f in os.listdir(sim_dir):
            fpath = os.path.join(sim_dir, f)
            if os.path.isfile(fpath):
                os.remove(fpath)
            elif os.path.isdir(fpath):
                shutil.rmtree(fpath)


def main():
    parser = argparse.ArgumentParser(description='Single-hospital fine-tuning experiment')
    parser.add_argument('-c', '--config_path', required=True, help='Config file path')
    parser.add_argument('--hospital-group', type=int, choices=[1, 2, 3], required=True,
                        help='Hospital group to process (1, 2, or 3)')
    parser.add_argument('--num-sims', type=int, default=10,
                        help='Number of simulations per hospital (default: 10)')
    parser.add_argument('--test-mode', action='store_true',
                        help='Test mode: 1 sim, 1 epoch, first 3 hospitals only')

    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    output_base = config['output_path']
    num_sims = args.num_sims

    if args.test_mode:
        num_sims = 1
        import utils.constants as test_constants
        test_constants.WORKER_NUM = 1

    print("=" * 70)
    print("SINGLE-HOSPITAL FINE-TUNING")
    print("=" * 70)
    print(f"Hospital group: {args.hospital_group}")
    print(f"Simulations per hospital: {num_sims}")
    print(f"Output base: {output_base}")
    print(f"Test mode: {args.test_mode}")
    print("=" * 70)

    # Load all hospital data using existing loader
    print("\nLoading hospital data...")
    all_data_loaders = get_hospital_data_annos_loaders(config)
    available_hospitals = [h for h in all_data_loaders.keys() if h != ALL_KEY]

    # Get hospitals for this group
    target_hospitals = HOSPITAL_GROUPS[args.hospital_group]
    hospitals_to_process = [h for h in target_hospitals if h in available_hospitals]

    if args.test_mode:
        hospitals_to_process = hospitals_to_process[:3]
        print(f"TEST MODE: {len(hospitals_to_process)} hospitals, {num_sims} sim, forced 1 epoch")

    missing = set(target_hospitals) - set(available_hospitals)
    if missing:
        print(f"WARNING: These hospitals from group {args.hospital_group} are not available: {missing}")

    print(f"\nProcessing {len(hospitals_to_process)} hospitals: {hospitals_to_process}")

    model_path = get_model_path(config)
    successful = 0
    failed = 0

    for hosp_idx, hospital_name in enumerate(hospitals_to_process):
        print(f"\n{'='*70}")
        print(f"HOSPITAL [{hosp_idx+1}/{len(hospitals_to_process)}]: {hospital_name}")
        print(f"{'='*70}")

        hospital_output_dir = os.path.join(output_base, hospital_name)
        os.makedirs(hospital_output_dir, exist_ok=True)

        for sim_idx in range(num_sims):
            sim_dir = os.path.join(hospital_output_dir, f"simulation_{sim_idx}")

            # Only skip if the final CSV exists (complete simulation)
            if is_simulation_complete(sim_dir):
                print(f"\n   Simulation {sim_idx} already complete, skipping")
                successful += 1
                continue

            # Clean up any partial artifacts from an incomplete run
            clean_incomplete_simulation(sim_dir)
            os.makedirs(sim_dir, exist_ok=True)

            print(f"\n   --- Simulation {sim_idx} ---")

            model = None
            try:
                # Set seeds for reproducibility (different seed per simulation)
                random.seed(sim_idx)
                np.random.seed(sim_idx)
                torch.manual_seed(sim_idx)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(sim_idx)

                # Load fresh pretrained model (no weight transfer)
                model = get_model(config, model_path, NAIVE_UPDATE, use_random_init=False)

                # Set up update_dict (identical structure to CL)
                update_dict = {
                    SIMULATION_IDX: sim_idx,
                    ITERATION_IDX_INIT: 0,
                    OUTPUT_PATH: sim_dir,
                    'hospital_name': hospital_name,  # For unique temp_models dirs
                }

                # In test mode, force 1 epoch (bypass cross-validation)
                if args.test_mode:
                    update_dict['num_epochs_per_hospital'] = 1

                # Prepare data loaders for this hospital
                hospital_data_loaders = {
                    hospital_name: all_data_loaders[hospital_name],
                }
                if ALL_KEY in all_data_loaders:
                    hospital_data_loaders[ALL_KEY] = all_data_loaders[ALL_KEY]

                # Use process_hospital_sequentially with a SINGLE hospital.
                # This calls perform_continual_learning() which does:
                #   find_epoch() -> train for best_epoch -> inference
                # Identical methodology to CL, just no weight transfer.
                results, skipped = process_hospital_sequentially(
                    model=model,
                    hospitals_order=[hospital_name],
                    data_annos_loaders=hospital_data_loaders,
                    update_dict=update_dict,
                    config=config,
                )

                if skipped:
                    print(f"   FAILED: {hospital_name} simulation {sim_idx}")
                    failed += 1
                    continue

                # Save results (same format as CL training_hospital_N.csv)
                if results:
                    results[0].to_csv(os.path.join(sim_dir, "training_hospital_0.csv"), index=False)

                # Save metadata
                metadata = {
                    'experiment_type': 'single_hospital_fine_tuning',
                    'hospital': hospital_name,
                    'simulation': sim_idx,
                    'num_predictions': len(results[0]) if results else 0,
                    'model_path': model_path,
                    'timestamp': datetime.now().isoformat(),
                }
                with open(os.path.join(sim_dir, "ft_metadata.json"), 'w') as f:
                    json.dump(metadata, f, indent=2)

                successful += 1
                print(f"   Simulation {sim_idx} completed successfully")

            except Exception as e:
                print(f"   ERROR in {hospital_name} sim {sim_idx}: {e}")
                import traceback
                traceback.print_exc()
                failed += 1

            finally:
                # Cleanup GPU memory
                del model
                gc.collect()
                torch.cuda.empty_cache()

        # Reload data loaders between hospitals to prevent in-place state
        # pollution carrying over from one hospital's run to the next.
        all_data_loaders = get_hospital_data_annos_loaders(config)

    print(f"\n{'='*70}")
    print(f"DONE: {successful} succeeded, {failed} failed")
    print(f"Results in: {output_base}")
    print(f"{'='*70}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
