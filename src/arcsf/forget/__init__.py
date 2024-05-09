from arcsf.forget.grad_ascent import GradientAscentForgetter
from arcsf.forget.grad_diff import GradientDifferenceForgetter
from arcsf.forget.idk import IDKForgetter
from arcsf.forget.kl import KLForgetter

FORGET_CLASSES = {
    "ascent": GradientAscentForgetter,
    "difference": GradientDifferenceForgetter,
    "idk": IDKForgetter,
    "kl": KLForgetter,
}
