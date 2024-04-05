"""
Adapted from TOFU: A Task of Fictitious Unlearning for LLMs, P. Maini, Z.
Feng, A. Schwarzschild, Z.C. Lipton, and J.Z. Kolter, 2024.
https://github.com/locuslab/tofu/blob/main/dataloader.py
"""

import torch
from torch.nn import CrossEntropyLoss
from torch.nn.functional import logsigmoid

from arcsf.forget.base import Forgetter


class DPOForgetter(Forgetter):
    """
    Forgetter with a DPO loss function
    """

    def __init__(self, *args, oracle_model=None, **kwargs):
        if oracle_model is None:
            raise RuntimeError("DPOForgetter requires an oracle_model.")
        self.oracle_model = oracle_model
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs, _, idk_inputs = inputs
        idk_input_ids, idk_labels, idk_attention_mask = idk_inputs
        forget_input_ids, forget_labels, forget_attention_mask = forget_inputs
        idk_outputs = model(
            idk_input_ids, labels=idk_labels, attention_mask=idk_attention_mask
        )
        forget_outputs = model(
            forget_input_ids, labels=forget_labels, attention_mask=forget_attention_mask
        )

        with torch.no_grad():
            idk_outputs_oracle = self.oracle_model(
                idk_input_ids, labels=idk_labels, attention_mask=idk_attention_mask
            )
            forget_outputs_oracle = self.oracle_model(
                forget_input_ids,
                labels=forget_labels,
                attention_mask=forget_attention_mask,
            )
            idk_logits_oracle = idk_outputs_oracle.logits
            forget_logits_oracle = forget_outputs_oracle.logits

        idk_loss_oracle = -1 * get_batch_loss(idk_logits_oracle, idk_labels)
        forget_loss_oracle = -1 * get_batch_loss(forget_logits_oracle, forget_labels)

        idk_loss_current = -1 * get_batch_loss(idk_outputs.logits, idk_labels)
        forget_loss_current = -1 * get_batch_loss(forget_outputs.logits, forget_labels)

        pi_logratios = idk_loss_current - forget_loss_current
        ref_logratios = idk_loss_oracle - forget_loss_oracle

        beta = 0.1
        loss = -logsigmoid(beta * (pi_logratios - ref_logratios)).mean()
        print(loss.item())
        loss = -pi_logratios.mean()
        loss = -idk_loss_current.mean()

        outputs = forget_outputs

        return (loss, outputs) if return_outputs else loss


def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = CrossEntropyLoss(ignore_index=-100, reduction="none")
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1, -2), shifted_labels).sum(dim=-1)

    return loss
