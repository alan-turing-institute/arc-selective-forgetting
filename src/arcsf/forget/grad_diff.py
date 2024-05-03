"""
Adapted from TOFU: A Task of Fictitious Unlearning for LLMs, P. Maini, Z.
Feng, A. Schwarzschild, Z.C. Lipton, and J.Z. Kolter, 2024.
https://github.com/locuslab/tofu/blob/main/dataloader.py
"""

from arcsf.forget.base import Forgetter


class GradientDifferenceForgetter(Forgetter):
    """
    Forgetter with a gradient difference loss function, which aims to maximise the loss
    of the model on the forget inputs and minimise the loss of the model on the retain
    inputs.
    """

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs, retain_inputs = inputs

        outputs = model(**forget_inputs)
        forget_loss = -1 * outputs.loss

        retain_outputs = model(**retain_inputs)
        retain_loss = retain_outputs.loss
        loss = forget_loss + retain_loss

        return (loss, outputs) if return_outputs else loss
