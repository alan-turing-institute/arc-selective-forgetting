from transformers import Trainer

from arcsf.forget.grad_ascent import GradientAscentForgetter
from arcsf.forget.grad_diff import GradientDifferenceForgetter
from arcsf.forget.idk import IDKForgetter
from arcsf.forget.kl import KLForgetter

# Dict for selecting trainer type
TRAINER_CLS_DICT = {
    "trainer": Trainer,
    "ascent": GradientAscentForgetter,
    "difference": GradientDifferenceForgetter,
    "idk": IDKForgetter,
    "kl": KLForgetter,
}
