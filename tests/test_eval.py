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
    correct_losses = torch.full((4, 5), 10)
    correct_losses[:, 0] = -1
    incorrect_losses = torch.full_like(correct_losses, 10)
    incorrect_losses[:, 1:] = -1
    print(correct_losses)
    print(incorrect_losses)
    correct_ratio = truth_ratio(correct_losses)
    print(correct_ratio)
    incorrect_ratio = truth_ratio(incorrect_losses)
    print(incorrect_ratio)

    # assert correct_ratio[0] == pytest.approx(0)
    # assert math.isinf(incorrect_ratio[0])
    raise NotImplementedError


def test_pipeline():
    raise NotImplementedError
