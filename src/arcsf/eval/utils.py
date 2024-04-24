from torch.nn import CrossEntropyLoss

_loss_function = CrossEntropyLoss(ignore_index=-100, reduction="none")


def get_loss(output_logits, labels):
    output_logits = output_logits[..., :-1, :].contiguous()
    shifted_labels = labels[..., 1:].contiguous()
    loss = _loss_function(output_logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
    return loss
