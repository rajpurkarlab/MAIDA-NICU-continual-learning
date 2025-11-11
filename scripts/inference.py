import os

import kwcoco
from utils.MAIDA_Dataset import MAIDA_Dataset
from utils.common_helpers import get_image_metadata
from utils.config_helpers import (
    get_model_path,
    get_output_path_for_inference,
)
from utils.common_helpers import get_model
from utils.AnnotationLoader import AnnotationLoader
from torch.utils.data import DataLoader

from utils.constants import *
from utils.inference_helpers import inference, save_predictions



def get_hospitals_dataloaders(config, annotation_loaders):
    data_path = config["data_path"]

    dataloaders = {}
    for split in [TRAIN_DATA_SOURCE, TEST_DATA_SOURCE]:
        image_meta = annotation_loaders[split].get_all_image_meta()
        print(image_meta)
        dataset = MAIDA_Dataset(
            data_path, HOSPITAL_DATA_SOURCE, image_meta
        )
        dataloaders[split] = DataLoader(dataset, 
                                        batch_size=BATCH_SIZE, 
                                        num_workers=WORKER_NUM,
                                        shuffle=True)

        print(
            f"Using {image_meta[ANNO_HOSPITAL_NAME_FIELD].unique()} with {len(dataset)} images for inference ..."
        )

    return dataloaders

def get_hospital_annotation_loaders(config):
    test_coco_annotations = kwcoco.CocoDataset(config['test_annos_path'])
    train_coco_annotations = kwcoco.CocoDataset(config['train_annos_path'])

    return {TRAIN_DATA_SOURCE: AnnotationLoader(train_coco_annotations), 
            TEST_DATA_SOURCE: AnnotationLoader(test_coco_annotations)}
    
def main(config):
    output_path = get_output_path_for_inference(config)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Get model
    pretrained_model_path = get_model_path(config)
    print(f"Loading model from {pretrained_model_path}")
    carinaNet_model = get_model(config, pretrained_model_path, "naive")

    if config[INFERENCE_DATASET] == USE_HOSPITLAS_DATASET_FOR_INFERENCE:
        annotation_loaders = get_hospital_annotation_loaders(config)
        dataloader = get_hospitals_dataloaders(config, annotation_loaders)
        for split in [TRAIN_DATA_SOURCE, TEST_DATA_SOURCE]:
            all_predictions = inference(carinaNet_model, dataloader[split], annotation_loaders[split])
            save_predictions(all_predictions, config, output_path, split)
    else:
        raise ValueError(f"Invalid dataset for inference: {config[INFERENCE_DATASET]}")

    

    
