import argparse
import functools
import gc
import os
import time
import unicodedata

import numpy as np
import pandas as pd
import yaml

import wandb
import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split
import random
from tqdm import tqdm
import torch.nn as nn
from torch.optim import Optimizer
from utils.common_helpers import get_all_hospital_data, get_hospital_data_annos_loaders, get_model, wandb_setup_metrics
from utils.constants import *

from models.ETTModel import ETTModel
from utils.AnnotationLoader import AnnotationLoader
from utils.fine_tune_helpers import calculate_loss_and_error, validate
from utils.hyper_tune_helpers import train_one_epoch
import utils.inference_helpers as inference_helpers
import utils.model_helpers as model_helpers

from utils.training_helper import find_epoch
from utils.utils import (
    get_annotations_by_image_id,
    get_center_coordinates_from_bbox,
    is_true,
    normalize_hospital_name,
)

import optuna
from typing import Dict
import torch
from tqdm import tqdm
import os

random.seed(SEED)
print("Initialize random seed")

# add argument config_path, short for -c, default = 'config.yaml'
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config_path")
parser.add_argument("-sf", "--suffix", default="")
parser.add_argument("-m", "--mode", choices=["tune", "inference"], default="tune")
parser.add_argument("-i", '--hospital_index', type=int, default=0)
parser.add_argument("--random-init", action="store_true", 
                    help="Use random initialization instead of pre-trained weights")
parser.add_argument("--num-simulations", type=int, default=None,
                    help="Override number of simulations from config file")
parser.add_argument("--output-suffix", type=str, default="",
                    help="Add suffix to output path (e.g., '_single_sim')")



def train(config: dict, data_annos_loader: dict, update_dict: dict) -> ETTModel:
    # Get a fresh copy of the off-the-shelf model
    use_random_init = config.get("use_random_init", False)
    carinaNet_model = get_model(config, 
                        os.path.join(CARINA_NET_OTS_MODEL_DIR, DEFAULT_MODEL_NAME),
                        NAIVE_UPDATE,
                        use_random_init=use_random_init)

    # Set the best hyperparameters
    carinaNet_model.optimizer = model_helpers.get_optimizer(carinaNet_model.model, LEARNING_RATE, update_dict['weight_decay'])
    carinaNet_model.scheduler = model_helpers.get_scheduler(carinaNet_model.optimizer, update_dict['max_lr'], PCT_START)
    
    # Use cross validation to find the optimal number of epochs
    best_epoch = find_epoch(carinaNet_model, data_annos_loader, update_dict)
    print(f'Best epoch: {best_epoch}')
    
    carinaNet_model = train_on_full_training_set(data_annos_loader, carinaNet_model, best_epoch)

    return carinaNet_model

def objective(trial, config: dict, data_annos_loader: dict, update_dict: dict) -> float:
    # Get hyperparameters to test
    max_lr = trial.suggest_float('max_lr', 0.00001, 0.001, log=True)
    weight_decay = trial.suggest_float('weight_decay', 0.01, 0.1, log=True)
    
    # Get a fresh copy of the off-the-shelf model
    use_random_init = config.get("use_random_init", False)
    carinaNet_model = get_model(config, 
                        os.path.join(CARINA_NET_OTS_MODEL_DIR, DEFAULT_MODEL_NAME),
                        NAIVE_UPDATE,
                        use_random_init=use_random_init)
    
    # Set the hyperparameters
    carinaNet_model.optimizer = model_helpers.get_optimizer(carinaNet_model.model, LEARNING_RATE, weight_decay)
    carinaNet_model.scheduler = model_helpers.get_scheduler(carinaNet_model.optimizer, max_lr, PCT_START)
   
    # Use cross validation to find the optimal number of epochs
    best_epoch, loss = find_epoch(carinaNet_model, data_annos_loader, update_dict, return_loss=True)
    print(f'Trial {trial.number} - Best epoch: {best_epoch} - Loss: {loss}')
    
    del carinaNet_model
    torch.cuda.empty_cache()
    gc.collect() 
    
    return loss

def train_with_tuning(config: dict, data_annos_loader: dict, update_dict: dict):
    # Create a study object and specify the direction is 'minimize'
    study = optuna.create_study(direction='minimize')
    
    # Run optimization for 20 trials
    study.optimize(lambda trial: objective(trial, config, data_annos_loader, update_dict), 
                  n_trials=20)
    
    # Get the best trial
    best_trial = study.best_trial
    
    print(f"\nBest trial:")
    print(f"  Value: {best_trial.value}")
    print(f"  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
        
    # Train final model with the best parameters
    use_random_init = config.get("use_random_init", False)
    carinaNet_model = get_model(config, 
                        os.path.join(CARINA_NET_OTS_MODEL_DIR, DEFAULT_MODEL_NAME),
                        NAIVE_UPDATE,
                        use_random_init=use_random_init)
    
    # Set the best hyperparameters
    carinaNet_model.optimizer = model_helpers.get_optimizer(carinaNet_model.model, LEARNING_RATE, best_trial.params['weight_decay'])
    carinaNet_model.scheduler = model_helpers.get_scheduler(carinaNet_model.optimizer, best_trial.params['max_lr'], PCT_START)
    
    # Train with the best parameters
    best_epoch = find_epoch(carinaNet_model, data_annos_loader, update_dict)
    
    carinaNet_model = train_on_full_training_set(data_annos_loader, carinaNet_model, best_epoch)

    return carinaNet_model, best_trial

def train_on_full_training_set(data_annos_loader, carinaNet_model, best_epoch):
    for epoch in tqdm(range(best_epoch)):
        for batch_idx, batch in enumerate(data_annos_loader[TRAIN_DATA_SOURCE][DATA_LOADERS_KEY]):
            images, image_ids = batch["image"], batch["image_id"].tolist()

            carinaNet_model.update_weight(
                images,
                image_ids,
                data_annos_loader[TRAIN_DATA_SOURCE][ANNOS_LOADER_KEY],
                update_dict={'print_loss': True}
            )

            if model_helpers.UPDATE_ON_BATCH:
                carinaNet_model.scheduler.step()
                
        if not model_helpers.UPDATE_ON_BATCH:
            carinaNet_model.scheduler.step()
    
    return carinaNet_model

def wandb_setup(config):
    wandb.config = {"dataset": config[DATA_PATH], "batch_size": BATCH_SIZE}
    wandb.init(
        # set the wandb project where this run will be logged
        project=config['wandb_project_name'],
        # mode="disabled",
    )
    wandb_setup_metrics()
    
def finetune(config):

    output_path = os.path.join(config[OUTPUT_PATH], FINE_TUNING)
    
    # Create different folders based on initialization method
    if config.get("use_random_init", False):
        output_path = os.path.join(output_path, "random_init")
        print(" Using random initialization - outputs will be saved to 'random_init' folder")
    else:
        output_path = os.path.join(output_path, "pretrained")
        print(" Using pre-trained weights - outputs will be saved to 'pretrained' folder")

    if is_true(config, USE_ALL_BUT_TARGET_HOSPITALS_ONLY):
        output_path = os.path.join(
            output_path, ALL_BUT_TARGET_HOSPITALS_ONLY_FINETUNED
        )
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        finetune_on_all_hospitals(config, output_path)

    elif is_true(config, USE_ALL_HOSPITALS):
        output_path = os.path.join(
            output_path, ALL_HOSPITALS_FINETUNED
        )
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        finetune_on_all_hospitals(config, output_path)
    else:
        output_path = os.path.join(output_path, TARGET_HOSPITAL_ONLY_FINETUNED)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        finetune_on_target_hospital(config, output_path)


def finetune_on_all_hospitals(config, output_path):
    def inference_on_test(config, dataloader, annotation_loader, prediction_path):
        finetuned_model = get_model(config, prediction_path)

        # Print average ETT and carina error
        predictions_dict = {}
        
        finetuned_model.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                images, image_ids = batch["image"], batch["image_id"].tolist()
                images_and_ids = zip(images, image_ids)

                predictions = finetuned_model.predict(images_and_ids, annotation_loader)
                predictions_dict.update(predictions)

        mean_loss_err = calculate_loss_and_error(predictions_dict, annotation_loader)
        print(f'carina error: {mean_loss_err[f"{ANNO_CAT_CARINA}{ERROR_SUFFIX}"]} | tip error: {mean_loss_err[f"{ANNO_CAT_TIP}{ERROR_SUFFIX}"]}')
        print(f'carina recall: {mean_loss_err[f"{ANNO_CAT_CARINA}{RECALL_SUFFIX}"]} | tip recall: {mean_loss_err[f"{ANNO_CAT_TIP}{RECALL_SUFFIX}"]}')

    dataloaders, annotation_loaders = get_all_hospital_data(config)
    suffix = '' if config[SUFFIX_KEY] is None or config[SUFFIX_KEY] == '' else f'{config[SUFFIX_KEY]}_'

    for sim_i in range(config[NUM_SIM]):
        print(f'*** Simulation {sim_i} ***')
        
        simulation_output_path = os.path.join(output_path, f"{suffix}simulation_{sim_i}")
        if not os.path.exists(simulation_output_path):
            os.makedirs(simulation_output_path)
        
        prediction_path = os.path.join(simulation_output_path, "finetuned_model.pth")
        
        # Get off-the-shelf model
        use_random_init = config.get("use_random_init", False)
        carinaNet_model = get_model(config, os.path.join(CARINA_NET_OTS_MODEL_DIR, DEFAULT_MODEL_NAME), use_random_init=use_random_init)

        # Start training
        best_loss = float("inf")
        best_epoch = 0
        early_stopping_counter = 0
        train_batch_step = 0

        for epoch_idx in range(100):
            print(f" * Epoch {epoch_idx}")
            train_batch_step = train_one_epoch(
                carinaNet_model.model,
                dataloaders[TRAIN_DATA_SOURCE],
                annotation_loaders[TRAIN_DATA_SOURCE],
                carinaNet_model.optimizer,
                carinaNet_model.scheduler,
                train_batch_step,
                epoch_idx
            )

            # validation
            val_loss = validate(
                carinaNet_model, 
                dataloaders[VAL_DATA_SOURCE], 
                annotation_loaders[VAL_DATA_SOURCE], 
                epoch_idx
            )
            
            print(f'Epoch {epoch_idx}: val_loss = {val_loss} | best_epoch: {best_epoch} | early_stopping_counter: {early_stopping_counter} | learning rate: {carinaNet_model.optimizer.param_groups[0]["lr"]}')
            
            # check if validation loss has improved
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch_idx  
                early_stopping_counter = 0
                print(f"Saving model to {prediction_path}")
                carinaNet_model.save_model(
                    os.path.join(prediction_path)
                )
            else:
                early_stopping_counter += 1

            # check if early stopping criteria is met
            if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered!")   
                if not WANDB_OFF:         
                    wandb.run.log(
                            {
                                "final_val_loss": val_loss,
                                "best_epoch": best_epoch,
                            }
                    )
                break


        print('Best epoch index:', best_epoch)
        
    # inference on the test data
    inference_on_test(config, dataloaders[TEST_DATA_SOURCE], annotation_loaders[TEST_DATA_SOURCE], prediction_path)

def finetune_on_target_hospital(config, output_path):
    print('Finetuning on target hospital only')
    data_annos_loader = get_hospital_data_annos_loaders(config)
    suffix = '' if config[SUFFIX_KEY] is None or config[SUFFIX_KEY] == '' else f'{config[SUFFIX_KEY]}'

    all_predictions = []
    for sim_i in range(config[NUM_SIM]):
        print(f'*** Simulation {sim_i} ***')
             
        simulation_output_path = os.path.join(output_path, f"{suffix}simulation_{sim_i}")
        if not os.path.exists(simulation_output_path):
            os.makedirs(simulation_output_path)
            
        update_dict = {
            OUTPUT_PATH: simulation_output_path,
        }
        for hospital_no, index_hospital in enumerate(data_annos_loader.keys()):
            if index_hospital != config['target_hospital']:
                print(index_hospital)
                continue
            
            if index_hospital == ALL_KEY:
                continue

            print(f"{hospital_no} - {index_hospital}")
            st = time.time()
            index_loader = data_annos_loader[index_hospital]
            # if best params already exist, read from file
            best_params_path = os.path.join(simulation_output_path, f"{index_hospital}_best_params.yaml")
            if True: # os.path.exists(best_params_path):
                # with open(best_params_path, 'r') as file:
                #     best_trial_params = yaml.load(file, Loader=yaml.FullLoader)
                #     print(f"Using best params: {best_trial_params}")
                #     update_dict.update(best_trial_params)
                update_dict.update({
                    'max_lr': MAX_LR,
                    'weight_decay': WEIGHT_DECAY,
                    'hospital_name': index_hospital,  # Add hospital name for unique temp dirs
                })
                finetuned_model = train(config, index_loader, update_dict)
            else:
                finetuned_model, best_trial = train_with_tuning(config, index_loader, update_dict)
                # Save the best params
                with open(best_params_path, 'w') as file:
                    yaml.dump(best_trial.params, file)

            finetuned_model.save_model(
                os.path.join(simulation_output_path, f"{index_hospital}.pth")
            )

            # inference on the test data
            hospital_predictions = inference_helpers.inference(finetuned_model, 
                                        index_loader[TEST_DATA_SOURCE][DATA_LOADERS_KEY], 
                                        index_loader[TEST_DATA_SOURCE][ANNOS_LOADER_KEY])

            hospital_predictions[HOSPITAL_NAME_FIELD] = index_hospital
            hospital_predictions[SIMULATION_FIELD] = sim_i
            
            all_predictions.append(hospital_predictions)

            print(f"  - Elapsed time: {time.time()-st:.4f}s")

        # Save results after every simulation
        all_predictions_per_sim = pd.concat(all_predictions)

        # Include hospital name in the prediction filename
        target_hospital = config['target_hospital']
        prediction_path = os.path.join(simulation_output_path, f'{FINETUNE_OUTPUT_FILENAME}-{target_hospital}.csv')
        print(f"Saving predictions to {prediction_path}")
        all_predictions_per_sim.to_csv(prediction_path)

def inference(config):
    '''
    Load the finetuned model of each hospital and run inference on that hospital's data
    '''
    output_path = os.path.join(config[OUTPUT_PATH], FINE_TUNING)
    
    # Handle different folders based on initialization method
    if config.get("use_random_init", False):
        output_path = os.path.join(output_path, "random_init")
        print(" Looking for random initialization models in 'random_init' folder")
    else:
        output_path = os.path.join(output_path, "pretrained")
        print(" Looking for pre-trained models in 'pretrained' folder")
    
    output_path = os.path.join(output_path, TARGET_HOSPITAL_ONLY_FINETUNED)
    
    data_annos_loader = get_hospital_data_annos_loaders(config)
    
    # Go through each simulation folder in output_path
    for sim_dir in os.listdir(output_path):
        all_predictions = []
        
        if not os.path.isdir(os.path.join(output_path, sim_dir)):
            continue

        # Go through each hospital in simulation folder
        for model_path in os.listdir(os.path.join(output_path, sim_dir)):
            if not model_path.endswith('.pth'):
                continue

            hospital_name = model_path.split('.')[0]
            model = get_model(config, os.path.join(output_path, sim_dir, model_path))

            hospital_loader = data_annos_loader[hospital_name]
            
            # Inference on the test data
            hospital_predictions = inference_helpers.inference(model, 
                                            hospital_loader[TEST_DATA_SOURCE][DATA_LOADERS_KEY], 
                                            hospital_loader[TEST_DATA_SOURCE][ANNOS_LOADER_KEY])

            hospital_predictions[HOSPITAL_NAME_FIELD] = hospital_name
            hospital_predictions[SIMULATION_FIELD] = sim_dir
            all_predictions.append(hospital_predictions)
    
        all_predictions = pd.concat(all_predictions)    
        prediction_file_path = os.path.join(output_path, sim_dir, f'{FINETUNE_OUTPUT_FILENAME}.csv')
        all_predictions.to_csv(prediction_file_path, index=False)
        print(f"Predictions saved to {prediction_file_path}")
        
    return

if __name__ == "__main__":
    args = parser.parse_args()
    # config_path = '/home/vir247/scratch/nicu/MAIDA-continual-learning/configs/fine_tune/hospitals/our_data/fine_tune_on_all.yaml'
    config_path = os.path.join(CONFIG_DIR, FINE_TUNE_DIR, args.config_path)
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    config[SUFFIX_KEY] = args.suffix
    # Only override target_hospital if not already set in config
    if 'target_hospital' not in config or config['target_hospital'] is None:
        config['target_hospital'] = HOSPITALS_LIST[args.hospital_index]

    # Handle random initialization flag
    if args.random_init:
        config["use_random_init"] = True
        print(" Using random initialization instead of pre-trained weights")
    else:
        config["use_random_init"] = False
        print(" Using pre-trained weights")

    # Handle number of simulations override
    if args.num_simulations is not None:
        original_sims = config.get(NUM_SIM, 1)
        config[NUM_SIM] = args.num_simulations
        print(f" Overriding simulations: {original_sims} → {args.num_simulations}")

    # Handle output path suffix
    if args.output_suffix:
        original_path = config[OUTPUT_PATH]
        config[OUTPUT_PATH] = original_path + args.output_suffix
        print(f" Modified output path: {original_path} → {config[OUTPUT_PATH]}")

    print(f"=== Config ===")
    print(config)
    
    config['wandb_project_name'] = "finetune-target-hospital"

    if not WANDB_OFF:
        wandb_setup(config)

    if args.mode == "tune":
        finetune(config)    
    else:
        inference(config)
