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

    assert correct_ratio[0] == pytest.approx(0)
    assert math.isinf(incorrect_ratio[0])


def _format(question, answer, dummy_tokenizer, max_length):
    encoded = dummy_tokenizer(
        question + answer,
        max_length=max_length,
        truncation=False,
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


def _get_perturbed(question_rows, n_perturbed, n_samples):
    gt_splits = [None] * len(question_rows)
    for row_id, row in enumerate(question_rows):
        split = row.split("?")
        gt_splits[row_id] = (
            split[0] + "?",
            split[1],
        )
    all_splits = [None] * len(question_rows)
    for row_id in range(n_samples):
        row_gt = gt_splits[row_id][1]
        perturbed_ids = [(row_id + 1 + i) % n_samples for i in range(n_perturbed)]
        perturbed_capitals = [
            gt_splits[p_id][1].split(" ")[-1] for p_id in perturbed_ids
        ]
        answer_string = row_gt.strip(row_gt.split(" ")[-1])
        perturbed_rows = [answer_string + p_capital for p_capital in perturbed_capitals]
        all_splits[row_id] = gt_splits[row_id] + tuple(perturbed_rows)
    return all_splits


# end-to-end test
def eval_end_to_end(dummy_base_model, dummy_tokenizer, dummy_train_data):
    max_length = 40
    n_samples = 3
    n_perturbed = 1
    question_rows = dummy_train_data["text"][: n_samples * 2 : 2]

    all_inp = _get_perturbed(question_rows, n_perturbed, n_samples)

    inp = torch.zeros((n_samples, n_perturbed + 1, max_length), dtype=int)
    targets = torch.zeros((n_samples, n_perturbed + 1, max_length), dtype=int)
    attention_masks = torch.zeros((n_samples, n_perturbed + 1, max_length), dtype=int)

    for row_id, row in enumerate(all_inp):
        for sample_n in range(n_perturbed + 1):
            q = row[0]
            a = row[1 + sample_n]
            (
                inp[row_id, sample_n],
                targets[row_id, sample_n],
                attention_masks[row_id, sample_n],
            ) = _format(q, a, dummy_tokenizer, max_length)

    all_losses = torch.zeros((n_samples, n_perturbed + 1))

    gt_batch = {
        "input_ids": inp[:, 0, :],
        "labels": targets[:, 0, :],
        "attention_mask": attention_masks[:, 0, :],
    }
    gt_dummy_model_output = dummy_base_model(**gt_batch)
    gt_loss = get_loss(gt_dummy_model_output.logits, targets[:, 0, :])

    all_losses[:, 0] = gt_loss

    for perturbed_index in range(1, n_perturbed + 1):
        p_batch = {
            "input_ids": inp[:, perturbed_index, :],
            "labels": targets[:, perturbed_index, :],
            "attention_mask": attention_masks[:, perturbed_index, :],
        }
        p_dummy_model_output = dummy_base_model(**p_batch)
        all_losses[:, perturbed_index] = get_loss(
            p_dummy_model_output.logits, targets[:, perturbed_index, :]
        )

    means = torch.mean(all_losses, dim=0)
    tr = truth_ratio(all_losses)
    # checks the model performs better on the ground truth
    assert torch.all(tr < 1).item()
    # checks truth ratio is less than 1
    assert (means[0] < torch.min(means[1:])).item()
