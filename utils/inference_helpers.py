"""Standalone inference: run a CarinaNet model over a dataloader and persist
the predictions.

:func:`inference` iterates a dataloader in batches, calls the model's
``predict`` method, and flattens the results into a single tidy DataFrame via
:func:`utils.common_helpers.format_results` -- attaching batch/patient indexing
metadata so the output schema matches the continual-learning CSVs.
:func:`save_predictions` writes that DataFrame to a conventionally named CSV
under the run's output directory. Inference here is evaluation-only: the
per-batch ``update_dict`` uses fixed simulation/iteration indices and no loss
is computed.
"""

from tqdm import tqdm
import pandas as pd
import os

import wandb
from models.CarinaNet.CarinaNetModel import CarinaNetModel
from utils.AnnotationLoader import AnnotationLoader
from torch.utils.data import DataLoader
from utils.common_helpers import format_results
from utils.constants import *
from utils.fine_tune_helpers import calculate_loss_and_error

def inference(
    carinaNet_model: CarinaNetModel,
    dataloader: DataLoader,
    annotation_loader: AnnotationLoader,
):
    """Run the model over every batch in a dataloader and collect predictions.

    Args:
        carinaNet_model: Model exposing a ``predict(images_and_ids)`` method.
        dataloader: Yields batches with ``image`` and ``image_id`` entries.
        annotation_loader: Supplies ground-truth annotations used by
            ``format_results`` to align predictions with labels.

    Returns:
        A DataFrame of all predictions in the standard CL output schema.
    """
    all_predictions = []
    # patient_idx is a running, dataset-wide counter advanced by batch_size each
    # iteration; it starts at -1 so the first batch begins effectively at 0.
    patient_idx = -1
    for batch_idx, batch in tqdm(enumerate(dataloader)):
        images, image_ids = batch["image"], batch["image_id"].tolist()
        images_and_ids = zip(images, image_ids)
        batch_size = len(images)

        ### Prediction
        update_dict = {
            PATIENT_IDX_INIT: patient_idx,
            BATCH_IDX: batch_idx,
            BATCH_SIZE: batch_size,
            SIMULATION_IDX: 0,
            ITERATION_IDX_INIT: 0,
        }
        predictions = carinaNet_model.predict(images_and_ids)
            
        all_predictions += format_results(
            predictions,
            annotation_loader,
            update_dict)
        
        patient_idx += batch_size

    # convert all_predictions to a dataframe
    all_predictions = pd.DataFrame(all_predictions)

    return all_predictions

def save_predictions(all_predictions, config, output_path, split=None):
    """Write inference predictions to a CSV under ``output_path``.

    Args:
        all_predictions: DataFrame of predictions to save.
        config: Run configuration; ``INFERENCE_DATASET`` selects the naming
            convention and must be the supported hospitals dataset.
        output_path: Directory the CSV is written into.
        split: Optional split label folded into the filename.

    Raises:
        ValueError: If ``config[INFERENCE_DATASET]`` is not a supported dataset.
    """
    # Save predictions
    if config[INFERENCE_DATASET] == USE_HOSPITLAS_DATASET_FOR_INFERENCE:
        split = "" if None else f"{split}_"
        file_path = os.path.join(
            output_path,
            f"ots_hospitals-{split}{INFERENCE_OUTPUT_FILENAME}",
        )
    else:
        raise ValueError(f"Invalid dataset for inference: {config[INFERENCE_DATASET]}")

    all_predictions.to_csv(file_path, index=False)
    print(f"Predictions saved to {file_path}")