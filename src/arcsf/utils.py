import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from arcsf.constants import EXPERIMENT_OUTPUT_DIR


def seed_everything(seed: int) -> None:
    """Set random seeds for torch, numpy, random, and python.

    Args:
        seed: Seed to set.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def make_output_dir(experiment_name: str, train_type: str, start_time: str) -> Path:
    """Create a directory for the output of an experiment.

    Args:
        experiment_name: Name of the experiment.
        train_type: Type of training (full, retain, or a forget type).
        start_time: Start time of the experiment run.

    Returns:
        Path to the created directory.
    """
    path = EXPERIMENT_OUTPUT_DIR / experiment_name / train_type / start_time
    os.makedirs(path)
    return path


def get_model_path(
    experiment_name: str, train_type: str, start_time: str | None = None
) -> Path:
    """Get the expected path to a saved model from an experiment:
        output/{experiment_name}/{train_type}/{start_time}

    Args:
        experiment_name: Name of the experiment.
        train_type: Type of training (full, retain, or a forget type).
        start_time: Start time of the experiment. If None, the latest experiment run
            will be selected.

    Returns:
        Path to the expected model directory.

    Raises:
        FileNotFoundError: If a matching directory does not exist.
    """
    parent_dir = Path(f"{EXPERIMENT_OUTPUT_DIR}/{experiment_name}/{train_type}")
    if not parent_dir.is_dir():
        raise FileNotFoundError(f"Directory {parent_dir} does not exist.")

    if start_time is not None:
        return parent_dir / start_time

    # get all directories with 20240528-105332-123456 format (only checks for numbers
    # not actual datetime validity)
    dirs = parent_dir.glob(f"{'[0-9]'*8}-{'[0-9]'*6}-{'[0-9]'*6}")
    dirs = [d for d in dirs if d.is_dir()]
    if len(dirs) == 0:
        raise FileNotFoundError(
            f"No directories matching expected datetime format found in {parent_dir}."
        )
    # max assumes datetimes in a alphabetically-sortable format
    return max(dirs)


def get_datetime_str() -> str:
    """
    Returns:
        The current datetime as a string in the format %Y%m%d-%H%M%S-%f, e.g.
        20240528-105332-123456 (yearmonthday-hourminutesseconds-milliseconds).
    """
    return datetime.strftime(datetime.now(), "%Y%m%d-%H%M%S-%f")


def get_device() -> torch.device:
    """Gets the best available device for pytorch to use.
    (According to: gpu -> mps -> cpu) Currently only works for one GPU.

    Returns:
        torch.device: available torch device
    """
    if torch.cuda.is_available():
        return torch.device("gpu")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
