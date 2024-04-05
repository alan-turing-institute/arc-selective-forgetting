"""
Adapted from TOFU: A Task of Fictitious Unlearning for LLMs, P. Maini, Z.
Feng, A. Schwarzschild, Z.C. Lipton, and J.Z. Kolter, 2024.
https://github.com/locuslab/tofu/blob/main/dataloader.py
"""

from arcsf.forget.base import Forgetter


class GradientAscentForgetter(Forgetter):
    """
    Forgetter with a gradient ascent loss function, which simply maximises the loss of
    the model on the forget inputs.
    """

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs, _, _ = inputs
        input_ids, labels, attention_mask = forget_inputs
        outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
        forget_loss = -1 * outputs.loss
        return (forget_loss, outputs) if return_outputs else forget_loss
