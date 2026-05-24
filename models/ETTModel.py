"""Base interface for ETT landmark-detection models.

Defines the common API used by the continual-learning pipeline (weight update,
prediction, checkpointing, optimizer/scheduler reset). ``CarinaNetModel`` is the
concrete implementation; this base dispatches the naive weight-update step.
"""

import torch
from utils.AnnotationLoader import AnnotationLoader
from utils.constants import NAIVE_UPDATE


class ETTModel:
    def __init__(self, update_method: str):
        self.model = None
        self.update_method = update_method

        # save the initial model weights
        self.initial_model_weights = None

    def reset_optimizer(self) -> None:
        pass

    def reset_scheduler(self) -> None:
        pass
    
    def save_model(self, model_path: str, states: dict = None) -> None:
        pass

    def predict(
        self, images_and_ids: zip, annotation_loader: AnnotationLoader = None
    ) -> dict:
        pass

    def update_weight(
        self,
        images: torch.Tensor,
        image_ids: list[int],
        annotation_loader: AnnotationLoader,
        update_dict: dict = {},  # optional per-update context
    ) -> None:
        annotations = [
            annotation_loader.get_annotations_by_image_id(image_id)
            for image_id in image_ids
        ]
        if self.update_method == NAIVE_UPDATE:
            self.naive_update(images, annotations, update_dict)
        else:
            raise Exception("Update method not supported")
