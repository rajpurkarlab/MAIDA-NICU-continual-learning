"""Low-level training-loop primitives for CarinaNet fine-tuning.

This module provides the building blocks used to train and evaluate the
landmark-localization model for a single epoch. ``train_one_epoch`` runs one
full pass of supervised updates over a dataloader (gradient clipping, optimizer
step, and optional per-batch vs. per-epoch scheduler stepping controlled by
``model_helpers.UPDATE_ON_BATCH``). ``validate`` runs inference over a held-out
loader and returns a scalar loss. ``calculate_loss_and_error`` aggregates the
per-image prediction dicts into mean classification/regression losses plus
per-category (tip/carina) localization error and recall. All Weights & Biases
logging is guarded by the ``WANDB_OFF`` flag so the primitives run unchanged
when logging is disabled.
"""

import torch
from tqdm import tqdm
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import wandb

import numpy as np
import warnings


from models.ETTModel import ETTModel
from utils.AnnotationLoader import AnnotationLoader
from utils.constants import *
import utils.model_helpers as model_helpers
from utils.utils import get_annotations_by_image_id, get_center_coordinates_from_bbox

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    annotation_loader: AnnotationLoader,
    optimizer: Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    train_batch_step: int,
    epoch_idx: int,
) -> int:
    """Run one full training pass over ``dataloader`` and update the model.

    For each batch the model produces classification and regression losses,
    which are averaged, summed, and backpropagated with gradient-norm clipping.
    The scheduler is stepped per batch or per epoch depending on
    ``model_helpers.UPDATE_ON_BATCH``.

    Args:
        model: The detection model being trained (in ``train()`` mode here).
        dataloader: Yields batches of images and image ids.
        annotation_loader: Resolves ground-truth annotations by image id.
        optimizer: Optimizer applied after each batch.
        scheduler: LR scheduler, stepped per batch or per epoch.
        train_batch_step: Running global batch counter (used for W&B logging).
        epoch_idx: Index of the current epoch (informational).

    Returns:
        int: The updated ``train_batch_step`` after processing all batches.
    """
    iters = len(dataloader)

    model.train()
    for batch_idx, batch in tqdm(enumerate(dataloader)):
        images, image_ids = batch["image"], batch["image_id"].tolist()
        # Gather the ground-truth annotations for every image in the batch.
        annotations = [
            annotation_loader.get_annotations_by_image_id(image_id)
            for image_id in image_ids
        ]
        optimizer.zero_grad()

        # Forward pass returns the two component losses; route to GPU if available.
        if CUDA_AVAILABLE:
            classification_loss, regression_loss = model(
                [images.cuda().float(), annotations]
            )
        else:
            classification_loss, regression_loss = model([images.float(), annotations])

        # Reduce each per-sample loss to a scalar; treat a RuntimeWarning
        # (e.g. mean of an empty tensor) as an error so it can be inspected.
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            try:
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
            except:
                raise

        loss = classification_loss + regression_loss
        if not WANDB_OFF:
            wandb.run.log(
                {
                    "train_batch_step": train_batch_step,
                    "train_classification_loss": classification_loss.item(),
                    "train_regression_loss": regression_loss.item(),
                    "train_loss": loss.item(),
                    "actual_learning_rate": scheduler.get_last_lr()[0],
                }
            )

        loss.backward()

        # Clip gradients to stabilize the small-batch fine-tuning updates.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

        optimizer.step()

        # Step the scheduler per batch when configured (e.g. OneCycleLR).
        if model_helpers.UPDATE_ON_BATCH:
            scheduler.step()

        train_batch_step += 1

        del classification_loss
        del regression_loss
        
        # Finish a batch
        
    
    # Finish an epoch: step the scheduler once per epoch when not stepping per batch.
    if not model_helpers.UPDATE_ON_BATCH:
        scheduler.step()

    return train_batch_step


def validate(
    carinaNet_model: ETTModel,
    dataloader: DataLoader,
    annotation_loader: AnnotationLoader,
    val_epoch_step: int,
) -> float:
    """Run inference over a validation loader and return its total loss.

    Predictions are gathered in eval/no-grad mode, then reduced to mean
    classification and regression losses (plus per-category error/recall for
    logging) via ``calculate_loss_and_error``.

    Args:
        carinaNet_model: The model wrapper exposing ``.predict`` and ``.model``.
        dataloader: Yields the validation batches.
        annotation_loader: Resolves ground-truth annotations for scoring.
        val_epoch_step: Validation step counter (used for W&B logging).

    Returns:
        float: ``classification_loss + regression_loss`` over the loader.
    """
    predictions_dict = {}

    carinaNet_model.model.eval()
    # No gradients needed during validation inference.
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images, image_ids = batch["image"], batch["image_id"].tolist()
            images_and_ids = zip(images, image_ids)

            # Accumulate predictions keyed by image id across all batches.
            predictions = carinaNet_model.predict(images_and_ids, annotation_loader)
            predictions_dict.update(predictions)

    mean_loss_err = calculate_loss_and_error(predictions_dict, annotation_loader)
    classification_loss = mean_loss_err[CLASSIFICATION_LOSS]
    regression_loss = mean_loss_err[REGRESSION_LOSS]

    loss = classification_loss + regression_loss
    if not WANDB_OFF:
        wandb.run.log(
            {   
                "val_epoch_step": val_epoch_step,
                "val_classification_loss": classification_loss,
                "val_regression_loss": regression_loss,
                f"{ANNO_CAT_TIP}{ERROR_SUFFIX}": mean_loss_err[f"{ANNO_CAT_TIP}{ERROR_SUFFIX}"],
                f"{ANNO_CAT_CARINA}{ERROR_SUFFIX}": mean_loss_err[f"{ANNO_CAT_CARINA}{ERROR_SUFFIX}"],
                f"{ANNO_CAT_TIP}{RECALL_SUFFIX}": mean_loss_err[f"{ANNO_CAT_TIP}{RECALL_SUFFIX}"],
                f"{ANNO_CAT_CARINA}{RECALL_SUFFIX}": mean_loss_err[f"{ANNO_CAT_CARINA}{RECALL_SUFFIX}"],
                "val_loss": loss,  
            },
            commit=False
        )

    return loss

def calculate_loss_and_error(
    predictions_dict: dict, annotation_loader: AnnotationLoader
) -> dict:
    """Aggregate per-image predictions into mean losses, errors, and recall.

    For every predicted image, the classification and regression losses are
    collected, and each predicted landmark is matched against the ground-truth
    annotation of the same COCO category. Matched landmarks contribute a
    Euclidean pixel error and count as a recall hit; categories with no matching
    prediction count as a recall miss. Empty metrics default to 1.

    Args:
        predictions_dict: Maps image id to a prediction dict containing the
            component losses and per-class predicted points.
        annotation_loader: Provides the COCO annotations for ground truth.

    Returns:
        dict: Mean values keyed by classification/regression loss and by
        per-category (tip/carina) error and recall.
    """
    loss_err = {
        CLASSIFICATION_LOSS: [],
        REGRESSION_LOSS: [],
        f"{ANNO_CAT_TIP}{ERROR_SUFFIX}": [],
        f"{ANNO_CAT_CARINA}{ERROR_SUFFIX}": [],
        f"{ANNO_CAT_TIP}{RECALL_SUFFIX}": [],
        f"{ANNO_CAT_CARINA}{RECALL_SUFFIX}": [],
    }

    coco_annotations = annotation_loader.coco_annotations

    for image_id, prediction in predictions_dict.items():
        # Collect the two component losses (tensors are converted to scalars).
        loss_err[CLASSIFICATION_LOSS].append(
            prediction[CLASSIFICATION_LOSS] if isinstance(prediction[CLASSIFICATION_LOSS], int) else prediction[CLASSIFICATION_LOSS].item())
            
        loss_err[REGRESSION_LOSS].append(
            prediction[REGRESSION_LOSS] if isinstance(prediction[REGRESSION_LOSS], int) else prediction[REGRESSION_LOSS].item())
       
        # Score each predicted landmark against its matching ground-truth point.
        for class_id, scores in prediction.items():
            # Skip prediction entries that are not detection categories.
            if not class_id in COCO_LABELS:
                continue

            coco_category_id = COCO_LABELS[class_id]
            coco_category_name = coco_annotations.cats[coco_category_id]["name"]

            coco_annots = get_annotations_by_image_id(coco_annotations, image_id)
            # select annotation whose category_id is coco_category_id
            for coco_annot in coco_annots:
                has_prediction = False
                if coco_annot["category_id"] == coco_category_id:
                    # Euclidean distance between ground-truth and predicted centers.
                    gPoint = get_center_coordinates_from_bbox(coco_annot["bbox"])
                    pPoint = scores["pred"]
                    error = np.sqrt(
                        (gPoint[0] - pPoint[0]) ** 2 + (gPoint[1] - pPoint[1]) ** 2
                    )
                    loss_err[f"{coco_category_name}{ERROR_SUFFIX}"].append(error)

                    # Matched landmark counts as a recall hit; stop at first match.
                    loss_err[f"{coco_category_name}{RECALL_SUFFIX}"].append(1)
                    has_prediction = True
                    break
            # No matching annotation found for this category: recall miss.
            if not has_prediction:
                loss_err[f"{coco_category_name}{RECALL_SUFFIX}"].append(0)

    # Average each metric; empty lists default to 1. Surface RuntimeWarnings.
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=RuntimeWarning)
        try:
            mean_loss_err = {k: 1 if len(v) == 0 else np.mean(v) for k, v in loss_err.items()}
        except:
            raise

    return mean_loss_err