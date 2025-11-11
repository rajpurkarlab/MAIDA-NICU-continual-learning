import argparse
import gc
from math import ceil
import os
import time

import torch
from torch.utils.data import DataLoader
import kwcoco
import json
import pandas as pd
import yaml
from models.CarinaNet.CarinaNetModel import CarinaNetModel
from models.ETTModel import ETTModel
from utils.CL_helpers import perform_continual_learning, perform_continual_learning_on_single_batch
from utils.AnnotationLoader import AnnotationLoader
from utils.MAIDA_Dataset import MAIDA_Dataset
# from utils.common_helpers import get_annotation_loader, get_hospital_data_annos_loaders, get_image_metadata, get_model 
from utils.common_helpers import get_hospital_data_annos_loaders, get_image_metadata, get_model 

from utils.config_helpers import get_model_path, get_output_path_for_global_CL
from utils.constants import *
import random
import wandb

from utils.utils import get_annotation_file_path, is_true, normalize_hospital_name, save_results
from typing import List

random.seed(SEED)
print(f"Initialize random seed {SEED}")

def process_hospital_sequentially(
    model: ETTModel,
    hospitals_order: list[str],
    data_annos_loaders: dict,
    update_dict: dict,
    config: dict = None
) -> tuple[List[pd.DataFrame], list]:
    """
    Perform continual learning on a list of hospitals sequentially, where all data from
    a hospital is used before moving to the next hospital.

    Returns:
        tuple: (results, skipped_hospitals)
            - results: List of dataframes, each contains predictions for a hospital's data
            - skipped_hospitals: List of dicts with info about hospitals that failed during training
    """

    # Check evaluation strategy from config
    eval_current_hospital_only = config.get(EVAL_CURRENT_HOSPITAL_ONLY, True)  # Default to True
    
    if eval_current_hospital_only:
        print(" DEPLOYMENT EVALUATION: Evaluating only on current hospital")
        # For deployment-focused evaluation, we'll set up the test loader for each hospital individually
        # This will be done inside the loop for each hospital
        index_loaders = {TEST_DATA_SOURCE: None, TRAIN_DATA_SOURCE: None}
    else:
        print(" RESEARCH EVALUATION: Evaluating on all hospitals")
        # Check if we're running a sanity check by looking for 'sanity_check' in config paths
        is_sanity_check = False
        if config and 'annos_dir' in config and 'sanity_check' in config['annos_dir']:
            is_sanity_check = True
            print(" SANITY CHECK MODE: Limiting inference to training hospitals only")
        
        if is_sanity_check:
            # For sanity check, create a limited test data loader with only the hospitals we're training on
            print(f" Sanity check: Inference will run on {len(hospitals_order)} hospitals instead of all hospitals")
            
            # Combine annotation loaders from the training hospitals
            combined_test_anno_loader = None
            combined_test_datasets = []
            
            for hospital in hospitals_order:
                if hospital in data_annos_loaders:
                    # Add annotation loader
                    if combined_test_anno_loader is None:
                        combined_test_anno_loader = data_annos_loaders[hospital][TEST_DATA_SOURCE][ANNOS_LOADER_KEY]
                    else:
                        combined_test_anno_loader = combined_test_anno_loader + data_annos_loaders[hospital][TEST_DATA_SOURCE][ANNOS_LOADER_KEY]
                    
                    # Add dataset
                    combined_test_datasets.append(data_annos_loaders[hospital][TEST_DATA_SOURCE][DATA_LOADERS_KEY].dataset)
            
            # Create a combined dataset
            from torch.utils.data import ConcatDataset
            combined_test_dataset = ConcatDataset(combined_test_datasets)
            combined_test_dataloader = DataLoader(combined_test_dataset, 
                                                batch_size=data_annos_loaders[hospitals_order[0]][TEST_DATA_SOURCE][DATA_LOADERS_KEY].batch_size,
                                                shuffle=False,
                                                num_workers=WORKER_NUM)
            
            # Set up the limited test loader
            index_loaders = {TEST_DATA_SOURCE: {DATA_LOADERS_KEY: combined_test_dataloader, 
                                              ANNOS_LOADER_KEY: combined_test_anno_loader},
                            TRAIN_DATA_SOURCE: None}
        else:
            # We perform inference on all hospitals (original behavior)
            index_loaders = {TEST_DATA_SOURCE: data_annos_loaders[ALL_KEY][TEST_DATA_SOURCE],
                              TRAIN_DATA_SOURCE: None}
        
    results = []
    skipped_hospitals = []
    successful_hospital_no = 0  # Track successful hospital number for model saving
    
    for hospital_no, index_hospital in enumerate(hospitals_order):
        print(f"{hospital_no} - {index_hospital}")
        st = time.time()
        
        try:
            # Check if hospital data exists
            if index_hospital not in data_annos_loaders:
                raise ValueError(f"Hospital {index_hospital} not found in data loaders")
            
            # We train on index hospital only
            index_loaders[TRAIN_DATA_SOURCE] = data_annos_loaders[index_hospital][TRAIN_DATA_SOURCE]
            
            # Set up test data based on evaluation strategy
            if eval_current_hospital_only:
                # For deployment evaluation: only test on current hospital
                index_loaders[TEST_DATA_SOURCE] = data_annos_loaders[index_hospital][TEST_DATA_SOURCE]
                print(f"   Evaluating on {index_hospital} only")
            # For research evaluation, test loader is already set up above

            all_predictions = perform_continual_learning(
                model,
                index_loaders,
                update_dict)

            # convert all_predictions to a dataframe
            all_predictions = pd.DataFrame(all_predictions)

            # iteration index is the index of a weight update step
            # Update initial iteration index for the next hospital
            update_dict[ITERATION_IDX_INIT] = all_predictions[ITERATION_FIELD].max() + 1

            print(f"  - Elapsed time: {time.time()-st:.4f}s")

            # Add hospital name to the dataframe
            all_predictions[INDEX_HOSPITAL_FIELD] = index_hospital
            all_predictions[SIMULATION_FIELD] = update_dict[SIMULATION_IDX]
            all_predictions[HOSPITAL_ORDER_FIELD] = successful_hospital_no  # Use successful hospital number

            results.append(all_predictions)

            # Save model after each hospital update
            model.save_model(
                os.path.join(update_dict[OUTPUT_PATH], f"{successful_hospital_no}.pth")
            )

            print(f"  - Elapsed time: {time.time()-st:.4f}s")
            print('Reset optimizer and scheduler')
            model.reset_optimizer()
            model.reset_scheduler()
            
            torch.cuda.empty_cache()
            
            # Increment successful hospital counter only if we reach here
            successful_hospital_no += 1
            
        except Exception as e:
            print(f"   ERROR processing {index_hospital}: {str(e)}")
            print(f"   STOPPING IMMEDIATELY - will not continue to remaining hospitals")

            # Add to skipped hospitals list
            skipped_hospitals.append({
                'hospital': index_hospital,
                'original_order': hospital_no,
                'error': str(e)
            })

            # Clean up any partial state
            torch.cuda.empty_cache()

            # Clean up temp_models directory
            import shutil
            if 'holdout_hospital' in update_dict:
                # Holdout analysis: use holdout-specific temp directory
                sim_idx = update_dict.get(SIMULATION_IDX, 0)
                temp_model_dir = os.path.join(update_dict[OUTPUT_PATH], f"temp_models_{update_dict['holdout_hospital']}_sim{sim_idx}")
            elif 'hospital_name' in update_dict:
                # Fine-tuning: use hospital-specific temp directory
                temp_model_dir = os.path.join(update_dict[OUTPUT_PATH], f"temp_models_{update_dict['hospital_name']}")
            else:
                # Regular CL: use standard temp_models directory
                temp_model_dir = os.path.join(update_dict[OUTPUT_PATH], "temp_models")

            if os.path.exists(temp_model_dir):
                print(f"    Cleaning up temp directory: {temp_model_dir}")
                shutil.rmtree(temp_model_dir)

            # Exit immediately instead of continuing
            print(f"   Exiting due to error - this simulation will be retried by the scheduler")
            break  # Exit the loop instead of continuing
    
    # Log skipped hospitals to the hospital order file
    if skipped_hospitals:
        hospital_order_file = os.path.join(update_dict[OUTPUT_PATH], "hospitals_order.txt")
        with open(hospital_order_file, "a") as f:
            f.write("\n\n# SKIPPED HOSPITALS (due to errors):\n")
            for skipped in skipped_hospitals:
                f.write(f"# SKIPPED: {skipped['hospital']} (original position {skipped['original_order']}) - Error: {skipped['error']}\n")
        
        print(f"\n  Summary: Skipped {len(skipped_hospitals)} hospitals due to errors:")
        for skipped in skipped_hospitals:
            print(f"  - {skipped['hospital']}: {skipped['error']}")
        print(f" Successfully processed {successful_hospital_no} hospitals")

    return results, skipped_hospitals

def main(config):
    
    output_path = get_output_path_for_global_CL(config)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    data_annos_loaders = get_hospital_data_annos_loaders(config)
    hospitals_order = list(data_annos_loaders.keys())
    hospitals_order = [hos for hos in hospitals_order if hos != ALL_KEY]

    model_path = get_model_path(config)
    
    for i in range(config[NUM_SIM]):
        if config[SKIP_SIMULATION] >= i:
            print(f"** Skip Simulation: {i}")
            continue

        print(f"** Simulation: {i}")

        # Create a subfolder for each simulation
        output_path_sim = os.path.join(output_path, f"simulation_{i}")
        if not os.path.exists(output_path_sim):
            os.makedirs(output_path_sim)

        # Shuffle hospitals order and save it to a txt file
        random.shuffle(hospitals_order)
        print(hospitals_order)
        with open(os.path.join(output_path_sim, "hospitals_order.txt"), "w") as f:
            f.write(",".join(hospitals_order))

        # Get model
        use_random_init = config.get("use_random_init", False)
        carinaNet_model = get_model(config, model_path, NAIVE_UPDATE, use_random_init=use_random_init)

        # initiate update_dict
        update_dict = {
            SIMULATION_IDX: i,
            ITERATION_IDX_INIT: 0,
            HAS_L2INIT: is_true(config, HAS_L2INIT),
            OUTPUT_PATH:output_path_sim,
        }
        
        # Add num_epochs_per_hospital if specified in config
        if 'num_epochs_per_hospital' in config:
            update_dict['num_epochs_per_hospital'] = config['num_epochs_per_hospital']
        
        if config[UPDATE_ORDER] == GLOBAL_SEQUENTIAL:
            results, skipped_hospitals = process_hospital_sequentially(
                carinaNet_model,
                hospitals_order,
                data_annos_loaders,
                update_dict,
                config,
            )
        else:
            raise ValueError(f"Invalid update order: {config[UPDATE_ORDER]}")

        # Save the latest results after each simulations
        save_results(results, output_path_sim, GLOBAL_CL, config)
    
        # Delete model at the end of the simulation
        del carinaNet_model
        gc.collect() 
        torch.cuda.empty_cache()

        ## Clean up after a simulation
        # 1. reset data and annotation loader any in case they get accidentally modified
        data_annos_loaders = get_hospital_data_annos_loaders(config)

        # 2. remove PREV_DATA from update_dict
        if PREV_DATA in update_dict:
            update_dict.pop(PREV_DATA, None)
