"""Cross-validation epoch selection for per-hospital continual-learning training.

This module decides how many epochs to fine-tune the model on a single
hospital's small (50-image) training set. ``find_epoch`` is the entry point: it
either honors a fixed epoch count passed via ``update_dict`` or runs K-fold
cross-validation, evaluating epochs incrementally and stopping early once
held-out loss stops improving. ``train_and_evaluate_for_single_epoch`` performs
one CV pass: it splits the training images into K folds, advances K per-fold
temp models by one epoch (resuming from disk checkpoints so folds accumulate
epochs across calls), and sums the held-out validation loss across folds.
``create_new_folds`` builds the per-fold train/val datasets and loaders, and
``get_model_and_checkpoint`` either resumes a fold's saved checkpoint or makes a
fresh copy of the base model with reset optimizer/scheduler. Temp-model
directory names are made unique per fine-tuning/holdout/CL context so parallel
runs do not collide.
"""

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
    """Select the number of fine-tuning epochs for one hospital.

    If ``update_dict['num_epochs_per_hospital']`` is set, that fixed count is
    returned immediately and cross-validation is skipped. Otherwise K-fold CV is
    run: the 50 training images are split into K folds, and at each candidate
    epoch ``train_and_evaluate_for_single_epoch`` advances every fold's temp
    model by one epoch and returns the summed held-out validation loss for that
    epoch. The epoch with the lowest total validation loss is tracked as
    ``best_epoch``; training stops early once the loss fails to improve for
    ``EARLY_STOPPING_PATIENCE`` consecutive epochs. The returned value is the
    epoch *count* (best epoch index + 1), not the index.

    Args:
        model: The base model to be fine-tuned; copied per fold, never mutated.
        loaders: Train data source loaders/annotation loaders keyed by constant.
        update_dict: Run configuration. May carry ``num_epochs_per_hospital``
            (fixed-epoch override), ``cv_folds``, ``OUTPUT_PATH``, and context
            keys (``hospital_name`` / ``holdout_hospital``) used to name the
            temp-model directory for parallel safety.
        return_loss: If True, also return the best (minimum) validation loss.

    Returns:
        int: The chosen number of epochs, or ``(epochs, best_loss)`` if
        ``return_loss`` is True.
    """

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
        
    # Evaluate increasing epoch counts; each iteration advances every CV fold by
    # one epoch and returns the summed held-out validation loss at that epoch.
    for epoch_idx in tqdm(range(50)):
        # Use hold-1-out cross validation to get the model performance after training fixed number of epochs
        total_loss = train_and_evaluate_for_single_epoch(model, loaders, update_dict, epoch_idx, temp_model_dir)
        print(f"Epoch {epoch_idx}: total_loss = {total_loss}")

        # Track the best epoch; reset the patience counter whenever loss improves.
        if total_loss < best_loss:
            best_loss = total_loss
            best_epoch = epoch_idx
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        print(f'epoch {epoch_idx} best_epoch: {best_epoch} early_stopping_counter: {early_stopping_counter}')
        # Stop once validation loss has not improved for PATIENCE epochs.
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
    """Advance every CV fold by one epoch and return summed held-out loss.

    The training images are partitioned into ``n_splits`` folds (leave-one-out
    if ``USE_HOLD_ONE_OUT_CV``, else ``cv_folds`` / default 10). For each fold a
    per-fold model is loaded from its on-disk checkpoint (resuming from prior
    epochs) or freshly copied for ``current_epoch == 0``, trained for one epoch
    on the fold's training split, re-saved, and evaluated on the fold's held-out
    split. The validation losses are summed across folds and normalized by the
    total image count. Assertions verify the folds collectively cover every
    training image exactly once.

    Args:
        model_orig: Base model that each fold copies; never mutated in place.
        loaders: Train data source loaders/annotation loaders keyed by constant.
        update_dict: Run configuration (passed through to ``update_weight``;
            may carry ``cv_folds``).
        current_epoch: Epoch index being evaluated; folds resume from the
            checkpoint saved at ``current_epoch - 1``.
        temp_model_dir: Directory holding the per-fold ``.pth`` checkpoints.

    Returns:
        float: Mean held-out validation loss across all folds for this epoch.
    """
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
    # Each fold: train its own model copy for one epoch, then score its held-out split.
    for fold_idx, (new_train_index, new_val_index) in enumerate(kf.split(train_image_meta)):
        new_train_dataloader, new_val_dataset = create_new_folds(train_dataset,
                                                                 train_image_meta, 
                                                                 new_train_index, 
                                                                 new_val_index)
        # Track held-out ids across folds to later assert full coverage.
        val_set_IDs = val_set_IDs.union(frozenset(new_val_dataset.get_image_meta()["id"]))
        # train + val splits must reconstruct the full training set, and the val
        # split size must be within one image of the even fold size.
        assert set(train_image_meta["id"]) == set(new_train_dataloader.dataset.image_ids + new_val_dataset.get_image_meta()["id"].to_list())
        assert len(new_val_dataset) >= len(train_image_meta) // n_splits 
        assert len(new_val_dataset) <= len(train_image_meta) // n_splits + 1

        # Use the fold index as the temporary model id
        temp_model_id = f'cv_fold_{fold_idx}'
        temp_model_path = os.path.join(temp_model_dir, f"{temp_model_id}.pth")

        model_copy, checkpoint = get_model_and_checkpoint(temp_model_path, model_orig)
        
        # We should use a fresh copy of the model for the first epoch
        if current_epoch == 0:
            assert checkpoint is None
                       
        # Train this fold's model for exactly one epoch over its train split.
        for batch_idx, batch in enumerate(new_train_dataloader):
            images, image_ids = batch["image"], batch["image_id"].tolist()

            if checkpoint is not None:
                # Resumed checkpoint must be exactly one epoch behind the current one.
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
        if not model_helpers.UPDATE_ON_BATCH:
            model_copy.scheduler.step()

        # Save the per-fold checkpoint so it can be resumed at the next epoch
        checkpoint = {
            'model_state_dict': model_copy.model.module.state_dict(),
            'optimizer': model_copy.optimizer.state_dict(),
            'scheduler': model_copy.scheduler.state_dict(),
            'epoch_idx': current_epoch
        }
        model_copy.save_model(temp_model_path, checkpoint)

        ### Perform inference on the held-out fold
        # Get all images and ids in the validation set
        images = [new_val_dataset[i]["image"] for i in range(len(new_val_dataset))]
        image_ids = [new_val_dataset[i]["image_id"] for i in range(len(new_val_dataset))]

        predictions = model_copy.predict(zip(images, image_ids), loaders[TRAIN_DATA_SOURCE][ANNOS_LOADER_KEY])

        mean_loss_err = calculate_loss_and_error(predictions, loaders[TRAIN_DATA_SOURCE][ANNOS_LOADER_KEY])
        classification_loss = mean_loss_err[CLASSIFICATION_LOSS]
        regression_loss = mean_loss_err[REGRESSION_LOSS]

        # Accumulate this fold's held-out loss into the epoch total.
        loss = classification_loss + regression_loss
        total_loss += loss

        # Free the per-fold model copy before moving to the next fold.
        del model_copy
        gc.collect()
        torch.cuda.empty_cache()

    # Validate that cross-validation covered all images (adaptive to dataset size)
    assert len(val_set_IDs) == train_image_count, f"Expected {train_image_count} images in validation set, got {len(val_set_IDs)}"
    assert set(train_image_meta["id"]) == val_set_IDs, "Cross-validation did not cover all training images"
    
    # Normalize the summed fold loss by the number of training images.
    total_loss = total_loss / train_image_count

    return total_loss

def create_new_folds(train_dataset, train_image_meta, new_train_index, new_val_index):
    """Build the train dataloader and validation dataset for one CV fold.

    Slices ``train_image_meta`` by the fold's train/val indices and wraps each
    slice in a fresh ``MAIDA_Dataset`` view over the same underlying dataset.

    Args:
        train_dataset: The full per-hospital training dataset.
        train_image_meta: Image metadata table for the full training set.
        new_train_index: Row indices for this fold's training images.
        new_val_index: Row indices for this fold's held-out validation images.

    Returns:
        tuple: ``(new_train_dataloader, new_val_dataset)`` for the fold.
    """
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
    Resume a fold's model from its checkpoint, or create a fresh copy.

    If a checkpoint exists at ``temp_model_path``, it is loaded and used to
    rebuild the model with its fine-tuned weights (no random init). Otherwise a
    deep copy of the base model is made with its optimizer and scheduler reset,
    so the original model is never mutated.

    Args:
        temp_model_path: Path to this fold's saved checkpoint, if any.
        model_orig: Base model to copy when no checkpoint exists.

    Returns:
        tuple: ``(model_copy, checkpoint)`` where ``checkpoint`` is ``None`` on
        a fresh copy.
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
