import math

import pytest
import torch

from arcsf.eval.metrics import (
    conditional_probability,
    eval_accuracy,
    ks_test,
    truth_ratio,
)
from arcsf.eval.utils import get_loss


def test_accuracy():
    # random predictions for test outputs
    test_outputs_correct = torch.randn(100, 5)
    # argmax predictions for test targets
    test_targets = test_outputs_correct.argmax(dim=1)
    # flip random predictions for incorrect predictions
    test_outputs_incorrect = test_outputs_correct * -1

    correct = eval_accuracy(test_outputs_correct, test_targets)
    assert correct["eval_accuracy"] == pytest.approx(1.0)

    incorrect = eval_accuracy(test_outputs_incorrect, test_targets)
    assert incorrect["eval_accuracy"] == pytest.approx(0.0)


def test_conditional_probability():
    correct_losses = torch.full((100, 5), 100)
    incorrect_losses = torch.full_like(correct_losses, 100)

    # correct answer is in the first index at evaluation time
    correct_losses[:, 0] = 0
    incorrect_losses[:, 1] = 0

    eval_prob = conditional_probability(correct_losses)
    assert torch.mean(eval_prob["conditional_probs"][0]).item() == pytest.approx(1.0)

    eval_prob = conditional_probability(incorrect_losses)
    assert torch.mean(eval_prob["conditional_probs"][0]).item() == pytest.approx(0.0)


def test_ks_test():
    probability_density, bin_edges = torch.histogram(
        torch.randn(1000), bins=10, density=True
    )
    bin_sizes = bin_edges[1:] - bin_edges[0:-1]
    test_cdf = torch.cumsum(probability_density * bin_sizes, dim=-1)
    comparison = ks_test(test_cdf, test_cdf)
    assert comparison[0] == pytest.approx(0.0)


def test_loss(dummy_tokenizer, dummy_forget_model):
    tokenizer = dummy_tokenizer
    model = dummy_forget_model

    test_input = "This is a test with a slightly longer string."

    tokenized_input = tokenizer(test_input, return_tensors="pt")
    input_ids = tokenized_input["input_ids"]

    attention_mask = tokenized_input["attention_mask"]

    labels = input_ids.clone()

    test_output = model(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels
    )

    loss = get_loss(test_output.logits, labels)

    assert isinstance(loss, torch.Tensor)
    assert loss.grad_fn


def test_truth_ratio():
    n_perturbed = 5
    correct_losses = torch.full((4, (n_perturbed + 1)), 10000)
    correct_losses[:, 0] = 0
    incorrect_losses = torch.full_like(correct_losses, 10000)
    incorrect_losses[:, 1:] = 0
    correct_ratio = truth_ratio(correct_losses)
    incorrect_ratio = truth_ratio(incorrect_losses)

    assert correct_ratio[0] == pytest.approx(1 / (n_perturbed + 1))
    assert math.isinf(incorrect_ratio[0])


def _format(question, answer, dummy_tokenizer, max_length):
    encoded = dummy_tokenizer(
        question + answer,
        max_length=max_length,
        truncation=True,
    )
    num_question_tokens = len(
        dummy_tokenizer.tokenize(question, add_special_tokens=True)
    )
    pad_length = max_length - len(encoded.input_ids)
    padded = encoded["input_ids"] + [dummy_tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded["attention_mask"] + [0] * pad_length
    label = encoded.input_ids
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = (
            encoded["input_ids"]
            + [dummy_tokenizer.eos_token_id]
            + [-100] * (pad_length - 1)
        )

    # change label to -100 for question tokens
    for i in range(num_question_tokens):
        label[i] = -100
    return (
        torch.tensor(padded),
        torch.tensor(label),
        torch.tensor(pad_attention_mask),
    )


# end-to-end test
def test_pipeline(dummy_base_model, dummy_tokenizer, dummy_train_data):
    max_length = 40
    n_samples = 3
    rows = dummy_train_data["text"][: n_samples * 2 : 2]

    inp = torch.zeros((n_samples, max_length), dtype=int)
    targets = torch.zeros((n_samples, max_length), dtype=int)
    attention_masks = torch.zeros((n_samples, max_length), dtype=int)

    for row_id, row in enumerate(rows):
        split = row.split("?")
        q, a = split[0] + "?", split[1]
        inp[row_id], targets[row_id], attention_masks[row_id] = _format(
            q, a, dummy_tokenizer, max_length
        )
    batch = {"input_ids": inp, "labels": targets, "attention_mask": attention_masks}
    dummy_model_output = dummy_base_model(**batch)
    loss = get_loss(dummy_model_output.logits, targets)
    # qualitative analysis use flag '-rP' in pytest
    for row in range(n_samples):
        row_target = targets[row][targets[row] != -100]
        print(dummy_tokenizer.decode(inp[row]).strip("<|endoftext|>"))
        print(dummy_tokenizer.decode(row_target))
        output = torch.argmax(dummy_model_output.logits[row], dim=-1)
        decoded_out = dummy_tokenizer.decode(output)
        print(decoded_out)
        print("\n")

    print(loss)
    raise NotImplementedError
