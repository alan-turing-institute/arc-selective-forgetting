"""
Adapted from TOFU: A Task of Fictitious Unlearning for LLMs, P. Maini, Z.
Feng, A. Schwarzschild, Z.C. Lipton, and J.Z. Kolter, 2024.
https://github.com/locuslab/tofu/blob/main/dataloader.py
"""

import torch
from torch import Tensor
from torch.nn.functional import kl_div, log_softmax
from transformers import BatchEncoding, PreTrainedModel
from transformers.utils.generic import ModelOutput

from arcsf.forget.base import Forgetter
from arcsf.models.model import load_maybe_peft_model


class KLForgetter(Forgetter):
    """
    Forgetter with a KL divergence loss function, which aims to maximise the loss of the
    model on the forget inputs, and minimise the KL divergence between the current
    model and the original (Oracle) model's logits on the retain inputs. See the parent
    Forgetter class for more information.

    Attributes:
        oracle_model: A copy of the original model (before unlearning) to compare
            the current model's logits to.
    """

    def __init__(self, *args, **kwargs):
        """
        Args:
            *args, **kwargs: Arguments passed to the parent Forgetter class (and via
                that to the HuggingFace Trainer).
        """
        super().__init__(*args, **kwargs)
        self.oracle_model = load_maybe_peft_model(
            self.model.config.name_or_path,
            merge=True,
            device_map="auto" if self.model.device.type != "cpu" else None,
        )
        self.oracle_model = self.accelerator.prepare(self.oracle_model)

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: tuple[BatchEncoding, BatchEncoding],
        return_outputs: bool = False,
    ) -> Tensor | tuple[Tensor, ModelOutput]:
        """See the parent Forgetter class for docs on expected inputs."""
        forget_inputs, retain_inputs = inputs

        outputs = model(**forget_inputs)
        forget_loss = -outputs.loss

        with torch.no_grad():
            retain_outputs = self.oracle_model(**retain_inputs)
        retain_probs = log_softmax(retain_outputs.logits, dim=-1)
        retain_probs = retain_probs.view(-1, retain_outputs.logits.shape[-1])

        current_outputs = model(**retain_inputs)
        current_probs = log_softmax(current_outputs.logits, dim=-1)
        current_probs = current_probs.view(-1, current_outputs.logits.shape[-1])

        retain_loss = kl_div(
            current_probs, retain_probs, reduction="batchmean", log_target=True
        )
        loss = forget_loss + retain_loss
        return (loss, outputs) if return_outputs else loss
