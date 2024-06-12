from pathlib import Path

from transformers import DataCollatorForLanguageModeling, Trainer

from arcsf.data.data_module import ForgetterDataCollator
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


# Dict for selecting trainer type and associated data collator
TRAINER_CLS_DICT = {
    "trainer": {
        "trainer_cls": Trainer,
        "data_collator": DataCollatorForLanguageModeling,
        "data_collator_kwargs": {"mlm": False},
    },
    "ascent": {
        "trainer_cls": GradientAscentForgetter,
        "data_collator": ForgetterDataCollator,
        "data_collator_kwargs": {},
    },
    "difference": {
        "trainer_cls": GradientDifferenceForgetter,
        "data_collator": ForgetterDataCollator,
        "data_collator_kwargs": {},
    },
    "idk": {
        "trainer_cls": IDKForgetter,
        "data_collator": ForgetterDataCollator,
        "data_collator_kwargs": {},
    },
    "kl": {
        "trainer_cls": KLForgetter,
        "data_collator": ForgetterDataCollator,
        "data_collator_kwargs": {},
    },
}
