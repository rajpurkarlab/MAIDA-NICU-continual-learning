"""Shared constants for the MAIDA-NICU continual learning pipeline.

Defines config field names, annotation/prediction column names, CarinaNet
hyperparameters, and the COCO label mapping used throughout the codebase.
These are wildcard-imported (``from utils.constants import *``) by the helper
modules, so every name defined here is part of the package's internal API.
"""

import random
import torch

CUDA_AVAILABLE = True
print(f"CUDA available: {CUDA_AVAILABLE}")

SEED = random.randint(0, 100000)
WANDB_OFF = True  # Disabled to allow running from different accounts without wandb authentication
FINE_TUNING = "fine_tune"

# The set of participating hospitals is derived at runtime from the annotation
# files present in the configured annotations directory, so no hospital list is
# hard-coded here.

# Expected directory and file names
CONFIG_DIR = "configs"
INFERENCE_DIR = "inference"
FINE_TUNE_DIR = "fine_tune"
CL_DIR = "CL"

#### Models
DEFAULT_MODEL_NAME = "model.pt"
CARINA_NET_OTS_MODEL_DIR = "models/CarinaNet"  # default model directory for CarinaNet
ALL_BUT_TARGET_HOSPITALS_ONLY_FINETUNED = "all-but-garget-hospitals-only"
ALL_HOSPITALS_FINETUNED = "all-hospitals"
TARGET_HOSPITAL_ONLY_FINETUNED = "target-hospital-only"
PUBLIC_ONLY_FINETUNED = "public-only"
TARGET_HOSPITAL_FINETUNED = "target-hospital"

### Metrics
CLASSIFICATION_LOSS = "classification_loss"
REGRESSION_LOSS = "regression_loss"
TOTAL_LOSS = "total_loss"
ERROR_SUFFIX = "-error"
RECALL_SUFFIX = "-recall"


#### Inference
INFERENCE_DATASET = "inference-dataset"
USE_PUBLIC_DATASET_FOR_INFERENCE = "public"
USE_HOSPITLAS_DATASET_FOR_INFERENCE = "hospitals"
USE_TARGET_HOSPITAL_DATASET_FOR_INFERENCE = "target-hospital"
INFERENCE_OUTPUT_FILENAME = "predictions.csv"

### Finetune
USE_ALL_BUT_TARGET_HOSPITALS_ONLY = "use_all-but-target-hospital_only"
USE_ALL_HOSPITALS = "use_all-hospitals"
FINETUNE_OUTPUT_FILENAME = "finetuned"

#### Continual Learning
GLOBAL_CL = "global_CL"
GLOBAL_HOSPITAL_MIX = "hospital-mix"
GLOBAL_SEQUENTIAL = "global_sequential"
SEQUENTIAL = "sequential"
HOSPITAL_SEQUENTIAL = "hospital_sequential"
NAIVE_UPDATE = "naive"

### Fields for update_dict
SIMULATION_IDX = "simulation_idx"
PATIENT_IDX_INIT = "patient_idx_init"
ITERATION_IDX_INIT = "iteration_idx_init"
BATCH_IDX = "batch_idx"

### Fields in config
UPDATE_ORDER = "update_order"  # CL update ordering (e.g. sequential)
MODEL_TYPE = "model_type"  # model variant identifier used in output filenames
UPDATE_METHOD = "update_method"  # continual-learning update method (naive)
MODEL_PATH = "model_path"
DATA_PATH = "data_path"
OUTPUT_PATH = "output_path"
TARGET_HOSPITAL = "target_hospital"
NUM_SIM = "number_of_simulation"
LEARNING_RATE = "learning_rate"
WEIGHT_DECAY_FIELD = "weight_decay"
T0_FIELD = "T_0"
EVAL_CURRENT_HOSPITAL_ONLY = "eval_current_hospital_only"  # Whether to evaluate only on current hospital vs all hospitals

WANDB_PROJECT_NAME = "wandb_project_name"


### Fields in the prediction dataframe
ITERATION_FIELD = "iteration"
HOSPITAL_NAME_FIELD = "hospital_name"
SIMULATION_FIELD = "simulation"
HOSPITAL_ORDER_FIELD = "hospital_order"
INDEX_HOSPITAL_FIELD = "index_hospital"

### We only support these data sources
HOSPITAL_DATA_SOURCE = "hospitals"
TEST_DATA_SOURCE = "test"
TRAIN_DATA_SOURCE = "train"
VAL_DATA_SOURCE = "val"

### Data
DATA_IMAGE_DIR = "images"
DATA_ANNOTATION_DIR = "annotations"
DATA_ANNOTATION_FILENAME = "annotations.json"

### Fields in the annotation file
ANNO_IMAGES_FIELD = "images"
ANNO_HOSPITAL_NAME_FIELD = "hospital_name"
ANNO_FILE_NAME_FIELD = "file_name"
ANNO_IMAGE_ID_FIELD = "id"
ANNO_CAT_TIP = "tip"
ANNO_CAT_CARINA = "carina"

### Hyperparameters for CarinaNet
LEARNING_RATE = 0.0000575198
WEIGHT_DECAY = 0.0379467	
MAX_LR = 0.000295155
PCT_START = 0.396094
TOTAL_STEP = 5000
T_0 = 10
NUM_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 6
WORKER_NUM = 4
BATCH_SIZE = 16
USE_HOLD_ONE_OUT_CV = False
AUGMENT_DATA = True
print(f'USE_HOLD_ONE_OUT_CV={USE_HOLD_ONE_OUT_CV}')
print(f'AUGMENT_DATA={AUGMENT_DATA}')

### COCO labels conversion

# Category ids in the NICU annotation files: 1 = ETT tip, 2 = carina
# (produced by preprocessing/convert_to_coco.py). They map to the model's
# two output classes via COCO_LABELS: model class 0 = tip, model class 1 = carina.
CAT_ID_TIP = 1
CAT_ID_CARINA = 2
COCO_LABELS = {0: CAT_ID_TIP, 1: CAT_ID_CARINA}
COCO_LABELS_INVERSE = {CAT_ID_TIP: 0, CAT_ID_CARINA: 1}

### Others
ANNOS_LOADER_KEY = "annos_loader"
DATA_LOADERS_KEY = "data_loaders"
SUFFIX_KEY = "suffix"
SKIP_SIMULATION = "skip_simulation"
ALL_KEY = "all"

### Evaluation
EVAL_CURRENT_HOSPITAL_ONLY = "eval_current_hospital_only"
