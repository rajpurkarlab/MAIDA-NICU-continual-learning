"""Data loading and result-formatting helpers shared across experiments.

This module centralizes the I/O and plumbing used by both the continual-learning
and fine-tuning entry points. Its responsibilities are threefold:

- Build PyTorch dataloaders from COCO-format annotation files. ``get_hospital_data_annos_loaders``
  scans the annotation directory, builds one train/test loader pair per hospital
  (excluding New-Somerset-Hospital, with a Singapore exclusion that does not match
  the current file naming) plus combined "ALL" loaders, while ``get_all_hospital_data``
  derives an internal train/validation split from the combined training set.
- Construct the model via ``get_model``, which currently always returns a
  ``CarinaNetModel`` (optionally loading pretrained weights).
- Convert raw per-image model output into the flat dictionary records written to
  the output CSVs (``format_results``), mapping the model's class ids to COCO
  category names and attaching ground-truth landmark coordinates and run metadata.

Dataloaders are constructed with ``shuffle=True`` and the shared ``WORKER_NUM`` /
``BATCH_SIZE`` constants. wandb metric definitions for training/validation are
registered by ``wandb_setup_metrics``.
"""

import json
import os
import kwcoco
import pandas as pd
import wandb
from torch.utils.data import DataLoader, RandomSampler

from models.CarinaNet.CarinaNetModel import CarinaNetModel
from models.ETTModel import ETTModel
from utils.AnnotationLoader import AnnotationLoader
from utils.MAIDA_Dataset import MAIDA_Dataset
from utils.constants import *
from utils.utils import (
    get_annotation_file_path,
    get_annotations_by_image_id,
    get_center_coordinates_from_bbox,
    get_image_filename_by_image_id,
    normalize_hospital_name,
)

def wandb_setup_metrics():
    """Register wandb metric definitions and their step axes.

    Training metrics are stepped by ``train_batch_step`` and validation metrics
    by ``val_epoch_step``, with each metric tagged with its optimization goal
    (minimize losses/errors, maximize recall) so wandb summaries pick the best value.
    """
    wandb.define_metric("train_batch_step")
    wandb.define_metric("train_classification_loss", step_metric="train_batch_step", step_sync=True, goal="minimize")
    wandb.define_metric("train_regression_loss", step_metric="train_batch_step", step_sync=True, goal="minimize")
    wandb.define_metric("weighted_tip_loss", step_metric="train_batch_step", step_sync=True, goal="minimize")
    wandb.define_metric("weighted_carina_loss", step_metric="train_batch_step", step_sync=True, goal="minimize")
    wandb.define_metric("train_loss", step_metric="train_batch_step", step_sync=True, goal="minimize")
    wandb.define_metric("actual_learning_rate", step_metric="train_batch_step", step_sync=True)

    wandb.define_metric("val_epoch_step")
    wandb.define_metric("val_classification_loss", step_metric="val_epoch_step", step_sync=True, goal="minimize")
    wandb.define_metric("val_regression_loss", step_metric="val_epoch_step", step_sync=True, goal="minimize")
    wandb.define_metric("val_loss", step_metric="val_epoch_step", step_sync=True, goal="minimize")
    wandb.define_metric(f"{ANNO_CAT_TIP}{ERROR_SUFFIX}", step_metric="val_epoch_step", step_sync=True, goal="minimize")
    wandb.define_metric(f"{ANNO_CAT_CARINA}{ERROR_SUFFIX}", step_metric="val_epoch_step", step_sync=True, goal="minimize")
    wandb.define_metric(f"{ANNO_CAT_TIP}{RECALL_SUFFIX}", step_metric="val_epoch_step", step_sync=True, goal="maximize")
    wandb.define_metric(f"{ANNO_CAT_CARINA}{RECALL_SUFFIX}", step_metric="val_epoch_step", step_sync=True, goal="maximize")
    wandb.define_metric("final_val_loss")
    wandb.define_metric("best_batch")
    
def get_image_metadata(annotation_file_path):
    """Load the ``images`` section of an annotation file as a DataFrame.

    Hospital names (when present) are normalized so they match the canonical
    naming used throughout the pipeline.

    Args:
        annotation_file_path: Path to a COCO-format annotation JSON file.

    Returns:
        A DataFrame of per-image metadata, with a normalized hospital-name column
        if one was present.
    """
    with open(annotation_file_path) as f:
        annotation_file = json.load(f)
    image_meta = pd.DataFrame(annotation_file[ANNO_IMAGES_FIELD])

    if ANNO_HOSPITAL_NAME_FIELD in image_meta.columns:
        image_meta[ANNO_HOSPITAL_NAME_FIELD] = image_meta[
            ANNO_HOSPITAL_NAME_FIELD
        ].apply(lambda x: normalize_hospital_name(x) if x is not None else None)
    
    return image_meta

def get_model(
    config: dict = None,
    model_path: str = None,
    update_method: str = None,
    use_random_init: bool = False,
) -> ETTModel:
    """
    Construct an ETTModel-derived model. Currently always returns a
    CarinaNetModel; weights are loaded from model_path when provided.
    """
    model = CarinaNetModel(model_path, update_method, use_random_init=use_random_init)

    return model


def format_results(
    predictions: dict,
    annotation_loader: AnnotationLoader,
    update_dict: dict,
) -> list[dict]:
    """Flatten raw model predictions into per-(image, landmark) CSV records.

    The model returns, per image, a dict keyed by class id. Only class ids present
    in ``COCO_LABELS`` are kept; each is mapped to its COCO category id and then to
    the category name ("tip"/"carina") via the loaded COCO annotations. For every
    kept landmark the function records the predicted scores plus the source file
    name, hospital name (when available), the matching ground-truth center point
    (``gPoint``, from the COCO bbox of the same category), and run metadata
    (iteration, simulation, batch size, patient index) drawn from ``update_dict``.

    Args:
        predictions: Mapping of image_id -> {class_id -> score dict} from the model.
        annotation_loader: Provides the COCO annotations and image metadata used to
            resolve file names, hospital names, and ground-truth coordinates.
        update_dict: Run-state values (indices, batch size) attached to each record.

    Returns:
        A list of dictionaries, one per (image, landmark) prediction, ready to be
        written as CSV rows.
    """
    ett_prediction_count = 0
    carina_prediction_count = 0
    all_predictions = []
    coco_annotations = annotation_loader.coco_annotations
    patient_idx = update_dict[PATIENT_IDX_INIT]
    
    # Get image metadata to access hospital names
    image_meta = annotation_loader.get_all_image_meta()

    # convert classification id to coco category name
    for image_id, prediction in predictions.items():
        file_name = get_image_filename_by_image_id(coco_annotations, image_id)
        
        # Get hospital name from image metadata
        hospital_name = None
        if ANNO_HOSPITAL_NAME_FIELD in image_meta.columns:
            hospital_row = image_meta[image_meta[ANNO_IMAGE_ID_FIELD] == image_id]
            if not hospital_row.empty:
                hospital_name = hospital_row.iloc[0][ANNO_HOSPITAL_NAME_FIELD]
        
        patient_idx += 1
        for class_id, scores in prediction.items():
            if not class_id in COCO_LABELS:
                continue
            
            scores["file_name"] = file_name
            
            # Add hospital name to predictions
            if hospital_name is not None:
                scores["hospital_name"] = hospital_name

            coco_category_id = COCO_LABELS[class_id]
            coco_category_name = coco_annotations.cats[coco_category_id]["name"]
            scores["category"] = coco_category_name

            coco_annots = get_annotations_by_image_id(coco_annotations, image_id)
            # select annotation whose category_id is coco_category_id
            for coco_annot in coco_annots:
                if coco_annot["category_id"] == coco_category_id:
                    scores["gPoint"] = get_center_coordinates_from_bbox(
                        coco_annot["bbox"]
                    )
                    break

            scores["iteration"] = update_dict[ITERATION_IDX_INIT] + update_dict[BATCH_IDX]
            scores["simulation"] = update_dict[SIMULATION_IDX]
            scores["batch_size"] = update_dict[BATCH_SIZE]
            scores["patient_idx"] = patient_idx

            all_predictions.append(scores)

            if coco_category_name == ANNO_CAT_TIP:
                ett_prediction_count += 1
            if coco_category_name == ANNO_CAT_CARINA:
                carina_prediction_count += 1

    return all_predictions

def get_all_hospital_data(config):
    """Build train/val/test loaders from the combined (all-hospital) dataset.

    Loads the combined ALL training set and deterministically (random_state=2024)
    carves it into a ~70% train / ~30% validation split, reusing the same underlying
    dataset via fresh image-metadata resets. The test loader is the combined ALL test
    set. Used for global hyperparameter tuning rather than per-hospital CL.

    Args:
        config: Experiment config passed through to the loader builders.

    Returns:
        A ``(dataloaders, annotation_loaders)`` tuple, each a dict keyed by
        TRAIN/VAL/TEST data-source constants. The train and val annotation loaders
        share the combined-train annotation loader.
    """
    # Tuning on the training sets of all hospitals
    data_annos_loader = get_hospital_data_annos_loaders(config)[ALL_KEY][TRAIN_DATA_SOURCE]
        
    # Randomly split the training set into 70% training and 30% validation
    train_dataset = data_annos_loader[DATA_LOADERS_KEY].dataset
    train_image_meta = train_dataset.get_image_meta()
    new_train_image_meta = train_image_meta.sample(frac=0.7, random_state=2024)
    new_val_image_meta = train_image_meta.drop(new_train_image_meta.index)
        
    new_train_dataset = MAIDA_Dataset(dataset=train_dataset)
    new_train_dataset.reset_image_meta(new_train_image_meta)
    new_train_dataloader = DataLoader(new_train_dataset, 
                                        num_workers=WORKER_NUM,
                                        batch_size=data_annos_loader[DATA_LOADERS_KEY].batch_size, 
                                        shuffle=True)
        
    new_val_dataset = MAIDA_Dataset(dataset=train_dataset)
    new_val_dataset.reset_image_meta(new_val_image_meta)
    new_val_dataloader = DataLoader(new_val_dataset, 
                                        num_workers=WORKER_NUM,
                                        batch_size=data_annos_loader[DATA_LOADERS_KEY].batch_size, 
                                        shuffle=True)
    
    assert len(new_val_dataset) + len(new_train_dataset) == len(train_dataset)
    
    test_data = get_hospital_data_annos_loaders(config)[ALL_KEY][TEST_DATA_SOURCE]
    dataloaders = {
            TRAIN_DATA_SOURCE: new_train_dataloader,
            VAL_DATA_SOURCE: new_val_dataloader,
            TEST_DATA_SOURCE: test_data[DATA_LOADERS_KEY]
        }
        
    annotation_loaders = {
            TRAIN_DATA_SOURCE: data_annos_loader[ANNOS_LOADER_KEY],
            VAL_DATA_SOURCE: data_annos_loader[ANNOS_LOADER_KEY],
            TEST_DATA_SOURCE: test_data[ANNOS_LOADER_KEY]
        }
    
    return dataloaders,annotation_loaders


def get_hospital_data_annos_loaders(config):
    """Build per-hospital and combined annotation/data loaders.

    Scans ``config['annos_dir']`` for ``{hospital}-{train,test}-annotations.json``
    files, skipping the combined ``hospital-*`` files (handled separately below) and
    excluded hospitals. For each remaining hospital it constructs an
    ``AnnotationLoader`` and a dataloader per split, then adds an ``ALL_KEY`` entry
    built from the combined train/test annotation paths in the config.

    Exclusions: New-Somerset-Hospital is dropped (low-resolution radiographs). The
    Singapore check matches the parenthesized name "National-University-(Singapore)",
    but the annotation files use "National-University-Singapore" (no parentheses), so
    Singapore is in practice retained.

    Args:
        config: Must provide ``annos_dir``, ``train_annos_path``, ``test_annos_path``.

    Returns:
        Nested dict: ``loaders[hospital][split] = {ANNOS_LOADER_KEY, DATA_LOADERS_KEY}``,
        plus a combined ``loaders[ALL_KEY][TRAIN/TEST]`` entry.
    """
    annos_dir = config['annos_dir']
    loaders = {}

    # go through every file in annos_dir
    for file in os.listdir(annos_dir):
        if file in ['hospital-test-annotations.json', 'hospital-train-annotations.json']:
            continue

        if TRAIN_DATA_SOURCE in file:
            split_type = TRAIN_DATA_SOURCE
        elif TEST_DATA_SOURCE in file:
            split_type = TEST_DATA_SOURCE
        else:
            print(f"Invalid file name: {file}")

        # get the hospital name from the file name
        hospital_name = file.split(f"-{split_type}-annotations.json")[0]
        
        # Excluded hospitals. New-Somerset-Hospital is dropped due to
        # low-resolution radiographs.

        if hospital_name == "New-Somerset-Hospital":
            continue

        if not hospital_name in loaders:
            loaders[hospital_name] = {}

        annos_loader = AnnotationLoader(kwcoco.CocoDataset(os.path.join(annos_dir, file)))
        dataloader = get_dataloader_from_annoloader(config, HOSPITAL_DATA_SOURCE, annos_loader)
        
        loaders[hospital_name][split_type] = {
            ANNOS_LOADER_KEY: annos_loader,
            DATA_LOADERS_KEY: dataloader
        }

    loaders[ALL_KEY] = {}
    
    all_train_annos_loader = AnnotationLoader(kwcoco.CocoDataset(config['train_annos_path']))
    all_train_dataloader = get_dataloader_from_annoloader(config, HOSPITAL_DATA_SOURCE, all_train_annos_loader)
    loaders[ALL_KEY][TRAIN_DATA_SOURCE] = {
        ANNOS_LOADER_KEY: all_train_annos_loader,
        DATA_LOADERS_KEY: all_train_dataloader
    }
    
    all_test_annos_loader = AnnotationLoader(kwcoco.CocoDataset(config['test_annos_path']))
    all_test_dataloader = get_dataloader_from_annoloader(config, HOSPITAL_DATA_SOURCE, all_test_annos_loader)
    loaders[ALL_KEY][TEST_DATA_SOURCE] = {
        ANNOS_LOADER_KEY: all_test_annos_loader,
        DATA_LOADERS_KEY: all_test_dataloader
    }

    return loaders

def get_dataloader_from_annoloader(config, data_source, annos_loader):
    """Wrap an AnnotationLoader's image metadata in a shuffled DataLoader.

    Args:
        config: Provides the image data root (``config[DATA_PATH]``).
        data_source: Data-source tag passed to ``MAIDA_Dataset``.
        annos_loader: AnnotationLoader supplying the image metadata.

    Returns:
        A DataLoader over the corresponding ``MAIDA_Dataset`` (shuffled, using the
        shared worker count and batch size).
    """
    dataset = MAIDA_Dataset(config[DATA_PATH], data_source, annos_loader.get_all_image_meta())
    dataloader = DataLoader(dataset, 
                            num_workers=WORKER_NUM,
                            batch_size=BATCH_SIZE, 
                            shuffle=True)
    return dataloader

