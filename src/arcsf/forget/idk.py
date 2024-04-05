"""
Adapted from TOFU: A Task of Fictitious Unlearning for LLMs, P. Maini, Z.
Feng, A. Schwarzschild, Z.C. Lipton, and J.Z. Kolter, 2024.
https://github.com/locuslab/tofu/blob/main/dataloader.py
"""

import torch

from arcsf.forget.base import Forgetter


class IDKForgetter(Forgetter):
    """
    Forgetter with a IDK (I don't know) objective, which aims to minimise the loss of
    the model on the retain data and the forget data with answers replaced with
    "I don't know"-like strings.
    """

    def compute_loss(self, model, inputs, return_outputs=False):
        _, retain_inputs, idk_inputs = inputs
        idk_input_ids, idk_labels, idk_attention_mask = idk_inputs
        retain_input_ids, retain_labels, retain_attention_mask = retain_inputs

        # concatenate the inputs. single forward pass is much more efficient
        input_ids = torch.cat((idk_input_ids, retain_input_ids), dim=0)
        labels = torch.cat((idk_labels, retain_labels), dim=0)
        attention_mask = torch.cat((idk_attention_mask, retain_attention_mask), dim=0)

        outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss
