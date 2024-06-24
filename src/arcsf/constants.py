from pathlib import Path

from transformers import Trainer

from arcsf.forget.grad_ascent import GradientAscentForgetter
from arcsf.forget.grad_diff import GradientDifferenceForgetter
from arcsf.forget.idk import IDKForgetter
from arcsf.forget.kl import KLForgetter

PROJECT_DIR = Path(__file__, "..", "..", "..").resolve()
CONFIG_DIR = Path(PROJECT_DIR, "configs")


def _get_config_dir(location):
    """Helper function for creating project config paths.

    Args:
        location: Directory inside PROJECT_ROOT/configs/ to create path for

    Returns:
        String giving path of PROJECT_ROOT/configs/location
    """
    return CONFIG_DIR / location


EXPERIMENT_CONFIG_DIR = _get_config_dir("experiment")
MODEL_CONFIG_DIR = _get_config_dir("model")
DATA_CONFIG_DIR = _get_config_dir("data")
FORGET_CONFIG_DIR = _get_config_dir("forget")

EXPERIMENT_OUTPUT_DIR = PROJECT_DIR / "output"


# Dict for selecting trainer type
TRAINER_CLS_DICT = {
    "trainer": Trainer,
    "ascent": GradientAscentForgetter,
    "difference": GradientDifferenceForgetter,
    "idk": IDKForgetter,
    "kl": KLForgetter,
}
