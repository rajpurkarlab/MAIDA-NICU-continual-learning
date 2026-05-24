"""Continual learning routines: train on a hospital, then run inference.

``perform_continual_learning`` selects an epoch count via cross-validation,
trains on the current hospital's data, and returns formatted test-set
predictions. Used by all CL experiment entry points.
"""

import copy
import pandas as pd
from tqdm import tqdm
import wandb

from models.CarinaNet.CarinaNetModel import CarinaNetModel
from models.ETTModel import ETTModel
from utils.AnnotationLoader import AnnotationLoader
from utils.MAIDA_Dataset import MAIDA_Dataset
from utils.common_helpers import format_results
from torch.utils.data import DataLoader, RandomSampler
from utils.constants import *
from utils.fine_tune_helpers import validate
from utils.hyper_tune_helpers import calculate_loss_and_error
import utils.model_helpers as model_helpers
from utils.training_helper import find_epoch


def perform_continual_learning(
    model: ETTModel,
    loaders: dict,
    update_dict: dict ,
) -> list[dict]:
    """
    Perform continual learning (first update then inference) with
    the given dataloader.
    """
    all_predictions = []
    patient_idx = -1
    
    ### perform training

    # Use cross validation to find the optimal number of epochs
    best_epoch = find_epoch(model, loaders, update_dict)
    
    iters = len(loaders[TRAIN_DATA_SOURCE][DATA_LOADERS_KEY])
    for epoch in range(best_epoch):
        for batch_idx, batch in enumerate(loaders[TRAIN_DATA_SOURCE][DATA_LOADERS_KEY]):
            images, image_ids = batch["image"], batch["image_id"].tolist()

            model.update_weight(
                images,
                image_ids,
                loaders[TRAIN_DATA_SOURCE][ANNOS_LOADER_KEY],
                update_dict,
            )
            
            if not WANDB_OFF:
                wandb.run.log(
                    {
                        "actual_learning_rate": model.scheduler.get_last_lr()[0],
                    }
                )
                
            if model_helpers.UPDATE_ON_BATCH:       
                model.scheduler.step()
        if not model_helpers.UPDATE_ON_BATCH:
            model.scheduler.step()

    ### Perform inference 
    for batch_idx, batch in tqdm(enumerate(loaders[TEST_DATA_SOURCE][DATA_LOADERS_KEY])):
        images, image_ids = batch["image"], batch["image_id"].tolist()
        batch_size = len(images)

        ### Predict
        predictions = model.predict(zip(images, image_ids))
        
        mean_loss_err = calculate_loss_and_error(predictions, loaders[TEST_DATA_SOURCE][ANNOS_LOADER_KEY])

        if not WANDB_OFF:
            wandb.run.log(
                {
                    f"{ANNO_CAT_TIP}{ERROR_SUFFIX}": mean_loss_err[f"{ANNO_CAT_TIP}{ERROR_SUFFIX}"],
                    f"{ANNO_CAT_CARINA}{ERROR_SUFFIX}": mean_loss_err[f"{ANNO_CAT_CARINA}{ERROR_SUFFIX}"],
                    f"{ANNO_CAT_TIP}{RECALL_SUFFIX}": mean_loss_err[f"{ANNO_CAT_TIP}{RECALL_SUFFIX}"],
                    f"{ANNO_CAT_CARINA}{RECALL_SUFFIX}": mean_loss_err[f"{ANNO_CAT_CARINA}{RECALL_SUFFIX}"],
                },
                commit=False
            )

        update_dict[PATIENT_IDX_INIT] = patient_idx
        update_dict[BATCH_IDX] = batch_idx
        update_dict[BATCH_SIZE] = batch_size

        all_predictions += format_results(
            predictions,
            loaders[TEST_DATA_SOURCE][ANNOS_LOADER_KEY],
            update_dict,
        )
        patient_idx += batch_size
    

    return all_predictions

def perform_continual_learning_on_single_batch(
    model: ETTModel,
    annotation_loader: AnnotationLoader,
    update_dict: dict ,
    batch: dict,) -> dict:
    """
    Perform inference then update on a given batch, and returns a dictionary of the predictions.
    """

    images, image_ids = batch["image"], batch["image_id"].tolist()
    
    ### Predict
    predictions = model.predict(zip(images, image_ids))
    batch_predictions = format_results(
        predictions,
        annotation_loader,
        update_dict,
    )

    ### Update
    model.update_weight(
        images,
        image_ids,
        annotation_loader,
        update_dict,
    )

    return batch_predictions