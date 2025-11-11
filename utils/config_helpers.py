import os
from utils.constants import *
from utils.utils import is_true

def get_model_path(config):
    # If model_path is specified, use it
    if "model_path" in config and (config["model_path"] != ""):
        return config["model_path"]

    # Return default CarinaNet OTS model
    return os.path.join(CARINA_NET_OTS_MODEL_DIR, DEFAULT_MODEL_NAME)

def get_output_path_for_inference(config):
    return os.path.join(config["output_path"], INFERENCE_DIR)

def get_output_path_for_global_CL(config):
    base_path = os.path.join(config["output_path"], GLOBAL_CL, config[UPDATE_METHOD])
    
    # Create different folders based on initialization method
    if config.get("use_random_init", False):
        init_path = os.path.join(base_path, "random_init")
    else:
        init_path = os.path.join(base_path, "pretrained")
    
    return os.path.join(init_path, config[SUFFIX_KEY])