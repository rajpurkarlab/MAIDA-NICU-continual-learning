"""Thin wrapper around a kwcoco ``CocoDataset`` that exposes image metadata
and ground-truth annotations in the shapes the training and inference loops
expect.

The class :class:`AnnotationLoader` caches the dataset's image records as a
pandas DataFrame (``image_meta``) for fast lookup, and converts per-image COCO
annotations into the tip/carina coordinate tensors consumed by CarinaNet via
:func:`utils.utils.convert_coco_annot_to_tensors`. Two datasets can be merged
with ``+`` (kwcoco union), and ``reset_image_meta`` re-points the loader at a
subset of images -- used when building cross-validation folds so the same
loaded annotations can be sliced without re-reading from disk.
"""

import os
from PIL import Image
import kwcoco
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from utils.constants import ANNO_FILE_NAME_FIELD, ANNO_IMAGE_ID_FIELD
from utils.utils import convert_coco_annot_to_tensors, get_annotations_by_image_id, get_image_file_path

class AnnotationLoader:
    def __init__(self, coco_annotations: kwcoco.CocoDataset):
        """Wrap a kwcoco dataset and cache its image records as a DataFrame.

        Args:
            coco_annotations: Loaded COCO-format dataset of images and annotations.
        """
        self.coco_annotations = coco_annotations

        # Cache the image records (one row per image) for fast metadata lookup
        # without repeatedly querying the kwcoco index.
        self.image_meta = pd.DataFrame(list(coco_annotations.imgs.values()))

    def __add__(self, other):
        """Merge two loaders into one via a kwcoco dataset union.

        Args:
            other: Another AnnotationLoader to combine with this one.

        Returns:
            A new AnnotationLoader covering the union of both datasets.
        """
        return AnnotationLoader(
            kwcoco.CocoDataset.union(self.coco_annotations, other.coco_annotations))

    def reset_image_meta(self, image_meta: pd.DataFrame):
        """Restrict the loader to the images present in ``image_meta``.

        Used to slice the loaded dataset down to a cross-validation fold without
        re-reading annotations from disk.

        Args:
            image_meta: Metadata DataFrame whose ``id`` column lists the image
                ids to keep.
        """
        self.image_meta = image_meta
        new_gids = list(self.image_meta["id"])

        # subset() keeps only the listed image ids and their annotations.
        self.coco_annotations = self.coco_annotations.subset(new_gids)

    def get_all_image_meta(self):
        """Return the cached image-metadata DataFrame for all loaded images."""
        return self.image_meta

    def get_annotations_by_image_id(self, image_id: str) -> torch.Tensor:
        """Return ground-truth tip/carina annotations for one image as a tensor.

        Args:
            image_id: COCO image id to fetch annotations for.

        Returns:
            Tensor of [x1, y1, x2, y2, label] rows for the image's tip/carina
            annotations (vertebra categories are dropped during conversion).
        """
        coco_annots = get_annotations_by_image_id(self.coco_annotations, image_id)
        return convert_coco_annot_to_tensors(coco_annots)