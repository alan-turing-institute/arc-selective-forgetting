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
        idk_inputs, retain_inputs = inputs

        # concatenate the inputs. single forward pass is more efficient
        input_ids = torch.cat(
            (idk_inputs["input_ids"], retain_inputs["input_ids"]), dim=0
        )
        labels = torch.cat((idk_inputs["labels"], retain_inputs["labels"]), dim=0)
        attention_mask = torch.cat(
            (idk_inputs["attention_mask"], retain_inputs["attention_mask"]), dim=0
        )

        outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
