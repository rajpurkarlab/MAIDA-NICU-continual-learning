import random
import torch

CUDA_AVAILABLE = True # CUDA_AVAILABLE
print(f"CUDA available: {CUDA_AVAILABLE}")

SEED = random.randint(0, 100000)
WANDB_OFF = True  # Disabled to allow running from different accounts without wandb authentication
FINE_TUNING = "fine_tune"
HYPERPARAMETER_TUNING = "hyperparameter_tuning"

HOSPITALS_LIST = [
'American-University-of-Beirut',
'Childrens-Hospital-Colorado',  # No apostrophe to avoid Unicode issues
'Chulalongkorn-University',
'Dr-Sardjito-Hospital',
'Fundacion-Santa-Fe-de-Bogota',  # No accents to avoid Unicode issues
'Indus',
'International-Islamic-Medical-University',
'Istanbul-Training-Research',
'King-Abdulaziz-Hospital',
'Kirikkale-Hospital-',  # Note: trailing dash in new annotations
'La-Paz-University-Hospital',
'Maharaj-Nakorn-Chiang-Mai-Hospital',  # NEW
'Medical-Center-of-South',
'National-Cheng-Kung-University-Hospital',
'National-University-Singapore',  # No parentheses
'Newark-Beth-Israel',
# 'New-Somerset-Hospital',  # NEW
'Osaka-Metropolitan-University',  # NEW
'Puerta-del-Mar-University-Hosptial',
'SES',
'Shiraz-University',
'Sichuan-People-Hospital',
'Sidra-Health',
'Tel-Aviv-Medical-Center',
'Tri-Service-General-Hospital',
'Uni-Tubingen',
'Universitaetsklinikum-Essen',  # NEW
'University-Hospital-Aachen',
'University-of-Graz',
'University-of-Kragujevac',
'University-of-Linz'
]

# Expected directory and file names
CONFIG_DIR = "configs"
INFERENCE_DIR = "inference"
FINE_TUNE_DIR = "fine_tune"
HYPER_TUNE_DIR = "hyperparameter_tuning"
CL_DIR = "CL"
SWEEP_CONFIG_FILENAME = "sweep_config.yaml"

#### Models
DEFAULT_MODEL_NAME = "model.pt"
CARINA_NET_OTS_MODEL_DIR = "models/CarinaNet"  # default model directory for CarinaNet
USE_CARINA_NET_HOSPITAL_FINETUNED = "hospital-specific"  # TODO remove
OFF_THE_SHELF = "OTS"
ALL_BUT_TARGET_HOSPITALS_ONLY_FINETUNED = "all-but-garget-hospitals-only"
ALL_HOSPITALS_FINETUNED = "all-hospitals"
TARGET_HOSPITAL_ONLY_FINETUNED = "target-hospital-only"
PUBLIC_ONLY_FINETUNED = "public-only"
PUBLIC_HOSPITALS_FINETUNED = "public-hospitals"
TARGET_HOSPITAL_FINETUNED = "target-hospital"

### Metrics
CLASSIFICATION_LOSS = "classification_loss"
REGRESSION_LOSS = "regression_loss"
TOTAL_LOSS = "total_loss"
ERROR_SUFFIX = "-error"
RECALL_SUFFIX = "-recall"


#### Inference
OTS_MODEL_INFERENCE = "OTS"
INFERENCE_DATASET = "inference-dataset"
USE_PUBLIC_DATASET_FOR_INFERENCE = "public"
USE_HOSPITLAS_DATASET_FOR_INFERENCE = "hospitals"
USE_TARGET_HOSPITAL_DATASET_FOR_INFERENCE = "target-hospital"
INFERENCE_OUTPUT_FILENAME = "predictions.csv"

### Finetune
USE_ALL_BUT_TARGET_HOSPITALS_ONLY = "use_all-but-target-hospital_only"
USE_ALL_HOSPITALS = "use_all-hospitals"
FINETUNE_OUTPUT_FILENAME = "finetuned"
NPLUS10 = "nplus10"

#### Continual Learning
INTRA_HOSPITAL_CL = "intra"
GLOBAL_CL = "global_CL"
GLOBAL_HOSPITAL_MIX = "hospital-mix"
GLOBAL_SEQUENTIAL = "global_sequential"
SEQUENTIAL = "sequential"
HOSPITAL_SEQUENTIAL = "hospital_sequential"
PATIENT_SEQUENTIAL = "patient_sequential"
BATCH_SEQUENTIAL = "batch_sequential"
PATIENT_BASED = "patient_based"
NAIVE_UPDATE = "naive"
EWC_UPDATE = "ewc"
REHEARSAL_UPDATE = "rehearsal"

### Fields for update_dict
SIMULATION_IDX = "simulation_idx"
PATIENT_IDX_INIT = "patient_idx_init"
ITERATION_IDX_INIT = "iteration_idx_init"
BATCH_IDX = "batch_idx"
PREV_DATA = "prev_data"
CURR_DOMAIN_NAME = "curr_domain_name"
CURR_DOMAIN_DATA = "curr_domain_data"
PREV_DOMAIN_NAME = "prev_domain_name"
IS_FINAL_UPDATE = "is_final_update"

### Fields in config
UPDATE_ORDER = "update_order" # choose among sequential, hospital mix, and full updates
MODEL_TYPE = "model_type" # choose among OTS, public-only, public-hospitals, target-hospital, and hospital-specific models
UPDATE_METHOD = "update_method" # choose among EWC, naive, and rehearsal
MODEL_PATH = "model_path"
DATA_PATH = "data_path"
OUTPUT_PATH = "output_path"
TARGET_HOSPITAL = "target_hospital"
NUM_SIM = "number_of_simulation"
IS_HYPER_TUNING = "is_hyperparameter_tuning"
LEARNING_RATE = "learning_rate"
WEIGHT_DECAY_FIELD = "weight_decay"
T0_FIELD = "T_0"
HAS_L2INIT = "has_L2_init"
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
EWC_LAMBDA = 1e6
NUM_EPOCHS = 10
L2_INIT_LAMBDA = 0.05
EARLY_STOPPING_PATIENCE = 6
WORKER_NUM = 4
BATCH_SIZE = 16
USE_HOLD_ONE_OUT_CV = False
AUGMENT_DATA = True
print(f'USE_HOLD_ONE_OUT_CV={USE_HOLD_ONE_OUT_CV}')
print(f'AUGMENT_DATA={AUGMENT_DATA}')

### COCO labels conversion

# To be consistent with CarinaNet implementation
# Reference
# "categories": [
    # {"id": 1, "name": "carina", "raw_id": 3046},
    # {"id": 2, "name": "tip", "raw_id": 3047}
# ]
CAT_ID_CARINA = 1
CAT_ID_TIP = 2
COCO_LABELS = {0:CAT_ID_CARINA, 1:CAT_ID_TIP}
COCO_LABELS_INVERSE = {CAT_ID_CARINA:0, CAT_ID_TIP:1}

### Others
ANNOS_LOADER_KEY = "annos_loader"
DATA_LOADERS_KEY = "data_loaders"
SUFFIX_KEY = "suffix"
SKIP_SIMULATION = "skip_simulation"
ALL_KEY = "all"

### New Constants for ICU+NICU Continual Learning
ICU_DATA_SOURCE = "icu"
NICU_DATA_SOURCE = "nicu"
ICU_NICU_DATA_SOURCE = "icu_nicu"

# ICU+NICU specific config keys
ICU_DATA_PATH = "icu_data_path"
ICU_ANNOS_DIR = "icu_annos_dir"
PASS_NICU_ONLY_WEIGHTS = "pass_nicu_only_weights"

# Data type identifiers for analysis
DATA_TYPE_ICU_NICU = "ICU+NICU"
DATA_TYPE_NICU_ONLY = "NICU-only"
DATA_TYPE_ICU_ONLY = "ICU-only"

# Training stage identifiers
STAGE_ICU = "ICU"
STAGE_NICU = "NICU"
STAGE_ICU_NICU = "ICU+NICU"

### Evaluation
EVAL_CURRENT_HOSPITAL_ONLY = "eval_current_hospital_only"
