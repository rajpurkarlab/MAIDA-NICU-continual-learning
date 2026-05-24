"""CarinaNet model wrapper (RetinaNet + ResNet-50 FPN) for ETT tip and carina detection.

Concrete ETTModel implementation. Builds the RetinaNet-based detector, either
loading pretrained CarinaNet weights or initializing a fresh COCO-pretrained
backbone (``use_random_init``), runs the naive training update (focal
classification + smooth-L1 regression, with the carina regression upweighted),
and at inference returns the single highest-confidence detection per landmark.
"""

from copy import deepcopy
import os
import random
import time

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import warnings
from models import CarinaNet
from models.CarinaNet import retinanet
from models.ETTModel import ETTModel
from utils.AnnotationLoader import AnnotationLoader
from utils.model_helpers import get_optimizer, get_scheduler
from utils.constants import *
import wandb

from utils.utils import is_true


class CarinaNetModel(ETTModel):
    def __init__(
        self,
        model_path: str,
        update_method: str,
        initial_model_weights: dict = None,
        checkpoint: dict = None,
        use_random_init: bool = False,
    ):
        super().__init__(update_method)

        torch.cuda.empty_cache()
        
        # Store the initialization method for later reference
        self._use_random_init = use_random_init
        
        if use_random_init:
            print("Using COCO pretrained RetinaNet weights (NO CarinaNet overlap)")
            # Create completely fresh RetinaNet model with COCO pretrained weights
            # DO NOT load any CarinaNet weights first
            from models.CarinaNet.retinanet import model as retinanet_model
            
            # Create fresh RetinaNet model with 2 classes (tip and carina)
            model = retinanet_model.resnet50(num_classes=2)
            # Use relative path from current file location
            current_dir = os.path.dirname(os.path.abspath(__file__))
            coco_weights_path = os.path.join(current_dir, "retinanet", "coco_resnet_50_map_0_335_state_dict.pt")
            
            # Load COCO weights and filter out incompatible layers
            coco_state_dict = torch.load(coco_weights_path, weights_only=False)
            
            # Remove classification output layers that have size mismatch (COCO has 80 classes, we have 2)
            keys_to_remove = [
                'classificationModel.output.weight',
                'classificationModel.output.bias'
            ]
            
            filtered_state_dict = {k: v for k, v in coco_state_dict.items() if k not in keys_to_remove}
            
            missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
            
            print(f"Loaded COCO weights - Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
            print("Classification output layers use fresh random initialization (2 classes vs COCO's 80)")
            print("NO CarinaNet weights loaded - pure COCO RetinaNet initialization")
            
            if CUDA_AVAILABLE:
                model = model.cuda()
            
            state_dict = {}  # Empty state dict for optimizer/scheduler loading
            
        else:
            # Load CarinaNet model architecture and then load checkpoint weights (silent)
            # Load the default CarinaNet model to get the model architecture
            if CUDA_AVAILABLE:
                model = torch.load(os.path.join(CARINA_NET_OTS_MODEL_DIR, DEFAULT_MODEL_NAME), weights_only=False).cuda().module
            else:
                model = torch.load(os.path.join(CARINA_NET_OTS_MODEL_DIR, DEFAULT_MODEL_NAME), map_location=torch.device("cpu"), weights_only=False).module
            
            # Load pre-trained weights
            if checkpoint is None:
                model_path = os.path.join(CARINA_NET_OTS_MODEL_DIR, DEFAULT_MODEL_NAME) if model_path == "" else model_path
                checkpoint = torch.load(model_path, weights_only=False)
            
            state_dict = checkpoint.state_dict() if isinstance(checkpoint, torch.nn.Module) else checkpoint
            model_state_dict = state_dict['model_state_dict'] if 'model_state_dict' in state_dict else state_dict
            
            model.load_state_dict({k.replace('module.', ''): v for k, v in model_state_dict.items()})

        model = torch.nn.DataParallel(model)                
        # print("Finished loading")

        # save the initial model weights
        self.initial_model_weights = deepcopy(model.state_dict()) if initial_model_weights is None else initial_model_weights

        self.model = model
        
        self.optimizer = get_optimizer(self.model, LEARNING_RATE, WEIGHT_DECAY)
        if not use_random_init and 'optimizer' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])
            
        self.scheduler = get_scheduler(self.optimizer, MAX_LR, PCT_START)
        if not use_random_init and 'scheduler' in state_dict:
            self.scheduler.load_state_dict(state_dict['scheduler'])

    def reset_optimizer(self) -> None:
        self.optimizer = get_optimizer(self.model, LEARNING_RATE, WEIGHT_DECAY)
        
    def reset_scheduler(self) -> None:
        self.scheduler = get_scheduler(self.optimizer, MAX_LR, PCT_START)

    def save_model(self, model_path: str, checkpoint: dict = None) -> None:
        if checkpoint:
            torch.save(
                checkpoint,
                model_path,
            )
            # print(f"Checkpoint saved to {model_path}")
        else:
            torch.save(self.model, model_path)
            # print(f"Model saved to {model_path}")

    def predict(
        self, images_and_ids: zip, annotation_loader: AnnotationLoader = None
    ) -> dict:
        """
        Perform inference on a batch of images. If annotation_loader is provided,
        the model will calculate the loss; otherwise, it will only perform inference.
        Return a dictionary of predictions per image.
        """
        predictions = {}

        self.model.eval()
        with torch.no_grad():
            for image, image_id in images_and_ids:
                predictions[image_id] = {}
                image = image.unsqueeze(0)
                image_input = image.cuda().float() if CUDA_AVAILABLE else image.float()

                if annotation_loader is not None:
                    annotation = annotation_loader.get_annotations_by_image_id(image_id)
                    inputs = [image_input, [annotation]]
                else:
                    inputs = image_input

                scores, classifications, transformed_anchors, focal_loss = self.model(
                    inputs
                )

                if focal_loss is not None:
                    classification_loss, regression_loss = focal_loss
                else:
                    classification_loss, regression_loss = -1, -1
                    
                predictions[image_id] = {
                        CLASSIFICATION_LOSS: classification_loss,
                        REGRESSION_LOSS: regression_loss,
                }

                scores = scores.cpu().numpy()
                classifications = classifications.cpu().numpy()

                idxs = np.array(
                    [
                        np.argmax(scores * (classifications == c))
                        for c in np.unique(classifications)
                    ]
                )  # Max detection of each class.
                
                for idx in idxs:
                    bbox = transformed_anchors[idx, :]
                    x1 = int(bbox[0])
                    y1 = int(bbox[1])
                    x2 = int(bbox[2])
                    y2 = int(bbox[3])
                    pred = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]

                    predictions[image_id][classifications[idx]] = {
                        "confidence": float(scores[idx]),
                        "pred": pred,
                    }

        return predictions

    def get_loss(self, images, annotations):
        """
        Helper function for the update functions. Return the loss in train mode
        on a batch of images and annotations.
        """
        self.model.train()
        # Note: self.model.module.freeze_bn() is called within the forward function instead

        if CUDA_AVAILABLE:
            classification_loss, regression_loss = self.model(
                [images.cuda().float(), annotations]
            )
        else:
            classification_loss, regression_loss = self.model(
                [images.float(), annotations]
            )

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
                    CLASSIFICATION_LOSS: classification_loss.item(),
                    REGRESSION_LOSS: regression_loss.item(),
                    TOTAL_LOSS: loss.item(),
                },
                commit=False,
            )
            
        return classification_loss, regression_loss, loss

    def naive_update(
        self, images: torch.Tensor, 
        annotations: list[torch.Tensor],
        naive_update_dict: dict,
    ) -> None:
        """
        Perform naive update on a batch of images and annotations.
        """
        self.optimizer.zero_grad()

        classification_loss, regression_loss, loss = self.get_loss(images, annotations)

        if not WANDB_OFF:
            wandb.run.log(
                {
                    CLASSIFICATION_LOSS: classification_loss.item(),
                    REGRESSION_LOSS: regression_loss.item(),
                    TOTAL_LOSS: loss.item(),
                },
                commit=False,
            )
            
        if is_true(naive_update_dict, 'print_loss'):
            print(f'Training loss: {loss.item()}')

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)

        self.optimizer.step()

        del classification_loss
        del regression_loss
