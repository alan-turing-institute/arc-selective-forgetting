"""
Adapted from TOFU: A Task of Fictitious Unlearning for LLMs, P. Maini, Z.
Feng, A. Schwarzschild, Z.C. Lipton, and J.Z. Kolter, 2024.
https://github.com/locuslab/tofu/blob/main/dataloader.py
"""

import torch
from torch.nn.functional import kl_div, log_softmax

from arcsf.forget.base import Forgetter


class KLForgetter(Forgetter):
    """
    Forgetter with a KL divergence loss function, which aims to maximise the loss of the
    model on the forget inputs, and minimise the KL divergence between the current
    model and the original (Oracle) model's logits on the retain inputs.
    """

    def __init__(self, *args, oracle_model=None, **kwargs):
        if oracle_model is None:
            raise RuntimeError("KLForgetter requires an oracle_model.")
        self.oracle_model = oracle_model
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs, retain_inputs, _ = inputs

        input_ids, labels, attention_mask = forget_inputs
        outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
        forget_loss = outputs.loss
        forget_loss = forget_loss * -1

        retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
        with torch.no_grad():
            retain_outputs = self.oracle_model(
                retain_input_ids,
                labels=retain_labels,
                attention_mask=retain_attention_mask,
            )
        retain_probs = log_softmax(retain_outputs.logits, dim=-1)
        retain_probs = retain_probs.view(-1, retain_outputs.logits.shape[-1])

        current_outputs = model(
            retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask
        )
        current_probs = log_softmax(current_outputs.logits, dim=-1)
        current_probs = current_probs.view(-1, current_outputs.logits.shape[-1])

        retain_loss = kl_div(
            current_probs, retain_probs, reduction="batchmean", log_target=True
        )
        loss = forget_loss + retain_loss
        return (loss, outputs) if return_outputs else loss
