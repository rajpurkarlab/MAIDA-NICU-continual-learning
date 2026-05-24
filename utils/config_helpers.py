"""Helpers for resolving model and output filesystem paths from a run config.

These functions centralize the path conventions used across experiments so the
entry-point scripts do not hard-code directory layouts. :func:`get_model_path`
chooses between an explicit checkpoint and the default off-the-shelf CarinaNet
weights; :func:`get_output_path_for_inference` and
:func:`get_output_path_for_global_CL` build the per-run output directories,
with the global-CL path further branched by update method, weight
initialization (random vs. pretrained), and a config-supplied suffix.
"""

import os
from utils.constants import *
from utils.utils import is_true

def get_model_path(config):
    """Resolve which model checkpoint to load for a run.

    Args:
        config: Run configuration dict.

    Returns:
        ``config["model_path"]`` if set to a non-empty value, otherwise the
        path to the default off-the-shelf (OTS) CarinaNet checkpoint.
    """
    # If model_path is specified, use it
    if "model_path" in config and (config["model_path"] != ""):
        return config["model_path"]

    # Return default CarinaNet OTS model
    return os.path.join(CARINA_NET_OTS_MODEL_DIR, DEFAULT_MODEL_NAME)

def get_output_path_for_inference(config):
    """Return the output directory for standalone inference runs.

    Args:
        config: Run configuration dict (must contain ``output_path``).

    Returns:
        ``<output_path>/<INFERENCE_DIR>``.
    """
    return os.path.join(config["output_path"], INFERENCE_DIR)

def get_output_path_for_global_CL(config):
    """Build the output directory for a global continual-learning run.

    The path encodes the update method, the weight-initialization strategy
    (random vs. pretrained), and a config-supplied suffix so that runs with
    different settings write to distinct directories.

    Args:
        config: Run configuration dict.

    Returns:
        ``<output_path>/<GLOBAL_CL>/<update_method>/<init>/<suffix>`` where
        ``<init>`` is ``random_init`` or ``pretrained``.
    """
    base_path = os.path.join(config["output_path"], GLOBAL_CL, config[UPDATE_METHOD])
    
    # Create different folders based on initialization method
    if config.get("use_random_init", False):
        init_path = os.path.join(base_path, "random_init")
    else:
        init_path = os.path.join(base_path, "pretrained")
    
    return os.path.join(init_path, config[SUFFIX_KEY])