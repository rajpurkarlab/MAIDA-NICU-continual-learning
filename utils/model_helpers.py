"""Optimizer and learning-rate scheduler factories for CarinaNet training.

Exposes :func:`get_optimizer` (an AdamW optimizer over all model parameters)
and :func:`get_scheduler` (a OneCycleLR schedule). Because OneCycleLR advances
once per optimizer step rather than per epoch, the module also publishes a
module-level flag ``UPDATE_ON_BATCH`` that the training loops read to decide
when to call ``scheduler.step()``. The flag is set as a side effect of
:func:`get_scheduler`, so the scheduler must be constructed before the flag is
consulted.
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer

from utils.constants import TOTAL_STEP

# Set by get_scheduler to indicate whether the scheduler steps per batch
# (OneCycleLR) or per epoch. Read by the training loops.
UPDATE_ON_BATCH = None

def get_optimizer(
    model: nn.Module, learning_rate: float, weight_decay: float
) -> Optimizer:
    """Build an AdamW optimizer over all of the model's parameters.

    Args:
        model: Model whose parameters will be optimized.
        learning_rate: Initial learning rate.
        weight_decay: Decoupled weight-decay coefficient.

    Returns:
        A configured AdamW optimizer.
    """
    return torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )


def get_scheduler(optimizer: Optimizer,
                  max_lr:float,
                  pct_start: float) -> torch.optim.lr_scheduler:
    """Build a OneCycleLR scheduler and flag it as a per-batch scheduler.

    Sets the module-level ``UPDATE_ON_BATCH`` to True so training loops know to
    step the scheduler after every batch (OneCycleLR's expected cadence) rather
    than once per epoch.

    Args:
        optimizer: Optimizer the schedule will modulate.
        max_lr: Peak learning rate reached during the cycle.
        pct_start: Fraction of total steps spent ramping up to ``max_lr``.

    Returns:
        A OneCycleLR scheduler spanning ``TOTAL_STEP`` total steps.
    """
    global UPDATE_ON_BATCH

    # OneCycleLR steps once per batch.
    UPDATE_ON_BATCH = True
    return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, pct_start=pct_start, total_steps=TOTAL_STEP)
