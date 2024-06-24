"""
Adapted from TOFU: A Task of Fictitious Unlearning for LLMs, P. Maini, Z.
Feng, A. Schwarzschild, Z.C. Lipton, and J.Z. Kolter, 2024.
https://github.com/locuslab/tofu/blob/main/dataloader.py
"""

from torch import Tensor
from transformers import BatchEncoding, PreTrainedModel
from transformers.utils.generic import ModelOutput

from arcsf.forget.base import Forgetter


class IDKForgetter(Forgetter):
    """
    Forgetter with a IDK (I don't know) objective, which aims to minimise the loss of
    the model on the retain data and the forget data with answers replaced with
    "I don't know"-like strings. See the parent Forgetter class for more information.
    """

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: tuple[BatchEncoding, BatchEncoding],
        return_outputs: bool = False,
    ) -> Tensor | tuple[Tensor, ModelOutput]:
        """See the parent Forgetter class for docs on expected inputs."""
        idk_inputs, retain_inputs = inputs

        idk_outputs = model(**idk_inputs)
        retain_outputs = model(**retain_inputs)
        loss = (idk_outputs.loss + retain_outputs.loss) / 2

        return (loss, idk_outputs) if return_outputs else loss
