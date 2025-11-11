import copy
import gc
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler
import wandb

from models.CarinaNet.CarinaNetModel import CarinaNetModel
from utils.MAIDA_Dataset import MAIDA_Dataset
from utils.constants import *
from utils.fine_tune_helpers import calculate_loss_and_error
import utils.model_helpers as model_helpers


def find_epoch(model, loaders, update_dict = {}, return_loss=False):    
    
    # Check if num_epochs_per_hospital is specified in update_dict - if so, skip cross-validation
    if 'num_epochs_per_hospital' in update_dict and update_dict['num_epochs_per_hospital'] is not None:
        fixed_epochs = update_dict['num_epochs_per_hospital']
        print(f"Using fixed epochs per hospital: {fixed_epochs} (skipping cross-validation)")
        
        if return_loss:
            return fixed_epochs, 0.0  # Return dummy loss value
        else:
            return fixed_epochs
    
    # Original cross-validation logic
    best_loss = float("inf")
    best_epoch = 0
    early_stopping_counter = 0
    
    # Check if we need unique temp dirs for parallel execution
    if 'hospital_name' in update_dict:
        # Fine-tuning: use hospital-specific temp directory
        temp_model_dir = os.path.join(update_dict[OUTPUT_PATH], f"temp_models_{update_dict['hospital_name']}")
    elif 'holdout_hospital' in update_dict:
        # Holdout analysis: use holdout-specific temp directory with simulation index for parallel runs
        sim_idx = update_dict.get(SIMULATION_IDX, 0)
        temp_model_dir = os.path.join(update_dict[OUTPUT_PATH], f"temp_models_{update_dict['holdout_hospital']}_sim{sim_idx}")
    else:
        # Regular CL: use standard temp_models directory
        temp_model_dir = os.path.join(update_dict[OUTPUT_PATH], "temp_models")

    # Create a temp_models directory if it does not exist, or clear the directory if it exists
    if not os.path.exists(temp_model_dir):
        os.makedirs(temp_model_dir)
    else:
        for file in os.listdir(temp_model_dir):
            os.remove(os.path.join(temp_model_dir, file))
        
    for epoch_idx in tqdm(range(50)):
        # Use hold-1-out cross validation to get the model performance after training fixed number of epochs
        total_loss = train_and_evaluate_for_single_epoch(model, loaders, update_dict, epoch_idx, temp_model_dir)
        print(f"Epoch {epoch_idx}: total_loss = {total_loss}")
        
        if total_loss < best_loss:
            best_loss = total_loss
            best_epoch = epoch_idx
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        print(f'epoch {epoch_idx} best_epoch: {best_epoch} early_stopping_counter: {early_stopping_counter}')
        if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
            break
        
    # Delete the temp_models
    for file in os.listdir(temp_model_dir):
        os.remove(os.path.join(temp_model_dir, file))
    os.rmdir(temp_model_dir)
    
    if not WANDB_OFF:
        wandb.run.log(
            {
                "total_loss": total_loss,
                "best_epoch": best_epoch,
            },
            commit=False
        )

    print(f"Best epoch index: {best_epoch}")

    best_epoch = best_epoch + 1  # returning the total number of epochs not the epoch index!
    
    if return_loss:
        return best_epoch, best_loss
    else:
        return best_epoch

def train_and_evaluate_for_single_epoch(model_orig, loaders, update_dict, current_epoch, temp_model_dir):
    train_dataset=loaders[TRAIN_DATA_SOURCE][DATA_LOADERS_KEY].dataset
    train_image_meta = train_dataset.get_image_meta()
    train_image_count = len(train_image_meta)
    
    # Validate that we have images for training (adaptive to dataset size)
    print(f"Training with {train_image_count} images for cross-validation")
    
    total_loss = 0
    
    if USE_HOLD_ONE_OUT_CV:
        n_splits = len(train_image_meta)
    else:
        # Check if cv_folds is specified in update_dict
        n_splits = update_dict.get('cv_folds', 10)  # Default to 10 if not specified

    # Create a dictionary from train_image_meta to keep track that all images are used
    # in the k-folds
    val_set_IDs = set()
    kf = KFold(n_splits=n_splits)
    for fold_idx, (new_train_index, new_val_index) in enumerate(kf.split(train_image_meta)):
        new_train_dataloader, new_val_dataset = create_new_folds(train_dataset,
                                                                 train_image_meta, 
                                                                 new_train_index, 
                                                                 new_val_index)
        val_set_IDs = val_set_IDs.union(frozenset(new_val_dataset.get_image_meta()["id"]))
        assert set(train_image_meta["id"]) == set(new_train_dataloader.dataset.image_ids + new_val_dataset.get_image_meta()["id"].to_list())
        assert len(new_val_dataset) >= len(train_image_meta) // n_splits 
        assert len(new_val_dataset) <= len(train_image_meta) // n_splits + 1
        
        # print(f'fold_idx: {i}, fold_size: {fold_size}')
        
        # let fold index be the model id
        temp_model_id = f'cv_fold_{fold_idx}'
        temp_model_path = os.path.join(temp_model_dir, f"{temp_model_id}.pth")

        model_copy, checkpoint = get_model_and_checkpoint(temp_model_path, model_orig)
        
        # We should use a fresh copy of the model for the first epoch
        if current_epoch == 0:
            assert checkpoint is None
                       
        model_updated = False            
        for batch_idx, batch in enumerate(new_train_dataloader):
            images, image_ids = batch["image"], batch["image_id"].tolist()
            
            if checkpoint is not None:
                # Make sure we are updating the model continuously
                assert (checkpoint['epoch_idx'] + 1) == current_epoch

            model_copy.update_weight(
                    images,
                    image_ids,
                    loaders[TRAIN_DATA_SOURCE][ANNOS_LOADER_KEY],
                    update_dict,
                )
            
            # Finish a single batch
            if model_helpers.UPDATE_ON_BATCH:
                model_copy.scheduler.step()
 
        # Finish a single epoch
        if model_updated & (not model_helpers.UPDATE_ON_BATCH):
            model_copy.scheduler.step()
            
        # print learning rate and loss
        # print(f"Epoch {current_epoch}, fold {fold_idx}, batch {batch_idx}, lr = {model_copy.scheduler.get_last_lr()[0]}")
        
        # save the model
        checkpoint = {
            'model_state_dict': model_copy.model.module.state_dict(),
            'optimizer': model_copy.optimizer.state_dict(),
            'scheduler': model_copy.scheduler.state_dict(),
            'epoch_idx': current_epoch
        }
        model_copy.save_model(temp_model_path, checkpoint)
        # print(f"Model saved at {temp_model_path}")
                                
        ### Perform inference on the held-out fold
        # Get all images and ids in the validation set
        images = [new_val_dataset[i]["image"] for i in range(len(new_val_dataset))]
        image_ids = [new_val_dataset[i]["image_id"] for i in range(len(new_val_dataset))]

        predictions = model_copy.predict(zip(images, image_ids), loaders[TRAIN_DATA_SOURCE][ANNOS_LOADER_KEY])

        mean_loss_err = calculate_loss_and_error(predictions, loaders[TRAIN_DATA_SOURCE][ANNOS_LOADER_KEY])
        classification_loss = mean_loss_err[CLASSIFICATION_LOSS]
        regression_loss = mean_loss_err[REGRESSION_LOSS]
        
        loss = classification_loss + regression_loss
        total_loss += loss

        del model_copy
        gc.collect()
        torch.cuda.empty_cache()

    # Validate that cross-validation covered all images (adaptive to dataset size)
    assert len(val_set_IDs) == train_image_count, f"Expected {train_image_count} images in validation set, got {len(val_set_IDs)}"
    assert set(train_image_meta["id"]) == val_set_IDs, "Cross-validation did not cover all training images"
    
    total_loss = total_loss / train_image_count
            
    return total_loss

def create_new_folds(train_dataset, train_image_meta, new_train_index, new_val_index):
    new_train_image_meta = train_image_meta.iloc[new_train_index]
    new_val_image_meta = train_image_meta.iloc[new_val_index]

    new_train_dataset = MAIDA_Dataset(dataset=train_dataset)
    new_train_dataset.reset_image_meta(new_train_image_meta)
    new_train_dataloader = DataLoader(new_train_dataset, 
                                        num_workers=WORKER_NUM,
                                        batch_size=BATCH_SIZE, 
                                        shuffle=True)

    new_val_dataset = MAIDA_Dataset(dataset=train_dataset)
    new_val_dataset.reset_image_meta(new_val_image_meta)
    
    return new_train_dataloader, new_val_dataset

def get_model_and_checkpoint(temp_model_path, model_orig):
    '''
    If temp_model_id is not in temp_model_dir, create one; otherwise, load the model
    '''
    checkpoint = None
    
    if os.path.exists(temp_model_path):
        checkpoint = torch.load(temp_model_path)

        # When loading from checkpoint, we want to load the fine-tuned weights
        # NOT start over with random initialization
        model_copy = CarinaNetModel(
            temp_model_path, 
            model_orig.update_method, 
            copy.deepcopy(model_orig.initial_model_weights), 
            checkpoint,
            use_random_init=False  # Always False when loading from checkpoint
        )
    else:
        # Make sure to use a copy of the model to avoid overwriting the weights
        model_copy = copy.deepcopy(model_orig)
        model_copy.reset_optimizer() # avoid accidentally modifying the original model's optimizer
        model_copy.reset_scheduler() # avoid accidentally modifying the original model's scheduler
    
    return model_copy, checkpoint
