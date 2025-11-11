import argparse
import os
import time
from datetime import datetime

import yaml
import torch.multiprocessing as mp

from utils.common_helpers import wandb_setup_metrics
from utils.constants import *
from continual_learning import main
import random
import wandb

from utils.utils import get_annotation_file_path, is_true, normalize_hospital_name, save_results

random.seed(SEED)
print("Initialize random seed")

# add argument config_path, short for -c, default = 'config.yaml'
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config_path", default="config.yaml")
parser.add_argument("-sk", "--skip_simulation", type=int, default=-1)
parser.add_argument("--random-init", action="store_true", 
                    help="Use random initialization instead of pre-trained weights")

# Default suffix should be today's date and a 8 digit random number in the 
# format of MM-DD-YY_XXXXXXXX

if __name__ == "__main__":
    args = parser.parse_args()
    config_path = os.path.join(CONFIG_DIR, CL_DIR, args.config_path)
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    # Generate current date in MM-DD-YY format and 8-digit random number
    current_date = datetime.now().strftime("%m-%d-%y")
    random_number = random.randint(10000000, 99999999)
    suffix = f"{current_date}_{random_number}"
    
    # Handle random initialization flag
    if args.random_init:
        config["use_random_init"] = True
        print(" Using random initialization instead of pre-trained weights")
        print(" Outputs will be saved to 'random_init' folder")
    else:
        config["use_random_init"] = False
        print(" Using pre-trained weights")
        print(" Outputs will be saved to 'pretrained' folder")
    
    config[SUFFIX_KEY] = suffix
    config[SKIP_SIMULATION] = args.skip_simulation

    config[UPDATE_ORDER] = GLOBAL_SEQUENTIAL

    # Check if wandb should be disabled from config
    use_wandb = not WANDB_OFF  # Use global constant as default
    if 'wandb_off' in config:
        use_wandb = not config['wandb_off']
    elif 'use_wandb' in config:
        use_wandb = config['use_wandb']

    if use_wandb:
        l2init = '_l2init' if is_true(config, HAS_L2INIT) else ''

        wandb.init(
            project=config.get('wandb_project_name', 'hospital-continual-learning'),
        )

        wandb_setup_metrics()
    else:
        print(" Wandb tracking disabled")

    main(config)

    if use_wandb:
        wandb.finish()
