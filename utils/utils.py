"""Low-level utilities shared across the continual-learning pipeline.

Groups three concerns: (1) filesystem path construction that encodes the
project's data layout (image paths swap ``.jpg`` for ``.png``; annotation
paths fall back to a default filename); (2) COCO annotation parsing, most
notably :func:`convert_coco_annot_to_tensors`, which keeps only tip (1) and
carina (2) categories, drops thoracic-vertebra categories, and converts boxes
from ``[x, y, w, h]`` to ``[x1, y1, x2, y2]``; and (3) small config and
result-saving helpers (:func:`is_true`, :func:`save_results`). Hospital names
are NFC-normalized so visually identical Unicode strings compare equal.
"""

import os
import unicodedata

import kwcoco
import numpy as np
import pandas as pd
from utils.constants import *


def is_true(config, field):
    """Return True only if ``field`` is present in ``config`` and truthy.

    Args:
        config: Configuration dict.
        field: Key to check.

    Returns:
        True if the key exists and its value is truthy, else False.
    """
    return (field in config) and (config[field])


def normalize_hospital_name(hospital_name):
    """NFC-normalize a hospital name so equivalent Unicode forms compare equal.

    Args:
        hospital_name: Raw hospital name string.

    Returns:
        The NFC-normalized string.
    """
    return unicodedata.normalize("NFC", hospital_name)


def get_annotation_file_path(
    data_dir: str,
    annotation_filename: str = None,
) -> str:
    """Build the path to an annotation file under the data directory.

    Args:
        data_dir: Root data directory.
        annotation_filename: Specific annotation filename; if omitted, the
            project default annotation filename is used.

    Returns:
        Full path to the annotation file.
    """
    if annotation_filename:
        return os.path.join(data_dir, DATA_ANNOTATION_DIR, annotation_filename)
    else:
        return os.path.join(data_dir, DATA_ANNOTATION_FILENAME)


def get_image_file_path(data_dir: str, data_source: str, file_name: str) -> str:
    """Build the on-disk path to a preprocessed image.

    Args:
        data_dir: Root data directory.
        data_source: Subdirectory for the image's source (e.g. hospital/split).
        file_name: Image filename from the annotations.

    Returns:
        Full path to the image, with the ``.jpg`` extension swapped for ``.png``
        since annotations reference ``.jpg`` but stored images are ``.png``.
    """
    return os.path.join(
        data_dir, DATA_IMAGE_DIR, data_source, file_name.replace(".jpg", ".png")
    )


def get_annotations_by_image_id(
    coco_annotations: kwcoco.CocoDataset, coco_image_id: int
) -> list[dict]:
    """Return the raw COCO annotation dicts belonging to one image.

    Args:
        coco_annotations: Loaded COCO dataset.
        coco_image_id: Image id to fetch annotations for.

    Returns:
        List of COCO annotation dicts for that image.
    """
    coco_anno_ids = coco_annotations.gid_to_aids[coco_image_id]
    coco_annos = [coco_annotations.anns[id] for id in coco_anno_ids]

    return coco_annos

def convert_coco_annot_to_tensors(coco_annots: list[dict]) -> np.ndarray:
    """Convert COCO annotation dicts into a tip/carina coordinate tensor.

    Keeps only tip (category 1) and carina (category 2) annotations, remaps
    their category ids to the model's label space, and converts each box from
    ``[x, y, w, h]`` to ``[x1, y1, x2, y2]``.

    Args:
        coco_annots: COCO annotation dicts for a single image.

    Returns:
        A tensor of shape (N, 5) with rows ``[x1, y1, x2, y2, label]``.
    """
    # get ground truth annotations
    annotations = np.zeros((0, 5))

    for coco_annot in coco_annots:
        # Skip annotations with category_id 3 and 4 (thoracic vertebra)
        # Only use tip (1) and carina (2) for training
        if coco_annot["category_id"] not in [1, 2]:
            continue
            
        annotation = np.zeros((1, 5))
        annotation[0, :4] = coco_annot["bbox"]
        annotation[0, 4] = COCO_LABELS_INVERSE[coco_annot["category_id"]]
        annotations = np.append(annotations, annotation, axis=0)

    # transform from [x, y, w, h] to [x1, y1, x2, y2]
    annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
    annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

    return torch.tensor(annotations)

def get_image_filename_by_image_id(
    coco_annotations: kwcoco.CocoDataset, coco_image_id: int
) -> str:
    """Return the stored filename for a COCO image id.

    Args:
        coco_annotations: Loaded COCO dataset.
        coco_image_id: Image id to look up.

    Returns:
        The image's filename as recorded in the annotations.
    """
    return coco_annotations.imgs[coco_image_id][ANNO_FILE_NAME_FIELD]


def get_center_coordinates_from_bbox(bbox: list[float]) -> tuple[float, float]:
    """Compute the center point of a ``[x, y, w, h]`` bounding box.

    Args:
        bbox: Box as ``[x, y, width, height]``.

    Returns:
        The ``(center_x, center_y)`` coordinates.
    """
    x, y, w, h = [float(v) for v in bbox]
    return x + w / 2, y + h / 2


def save_results(
    results: list[pd.DataFrame],
    output_path: str,
    cl_context: str,
    config: dict,
) -> None:
    """Concatenate per-step result frames and write them to a single CSV.

    Args:
        results: List of per-step/per-hospital result DataFrames.
        output_path: Directory the combined CSV is written into.
        cl_context: Context label folded into the filename.
        config: Run configuration; supplies model type, update order, and
            update method used to build the filename.
    """
    results = pd.concat(results, ignore_index=True)
    output_file_path = os.path.join(
        output_path,
        f'{config["model_type"]}_{cl_context}_{config[UPDATE_ORDER]}_{config[UPDATE_METHOD]}.csv',
    )

    results.to_csv(output_file_path, index=False)
    print(f"Results saved to {output_file_path}")
