import pytest
import torch

from arcsf.eval.metrics import eval_accuracy, eval_probability


def test_accuracy():
    # random predictions for test outputs
    test_outputs_correct = torch.randn(100, 10)
    # argmax predictions for test targets
    test_targets = test_outputs_correct.argmax(dim=1)
    # flip random predictions for incorrect predictions
    test_outputs_incorrect = test_outputs_correct * -1

    correct = eval_accuracy(test_outputs_correct, test_targets)
    assert correct["eval_accuracy"] == 1.0

    incorrect = eval_accuracy(test_outputs_incorrect, test_targets)
    assert incorrect["eval_accuracy"] == 0.0


def test_probability():
    correct_probs = torch.ones(10)
    correct_probs[0] = 100
    incorrect_probs = torch.ones_like(correct_probs)
    incorrect_probs[1] = 100

    eval_prob = eval_probability(correct_probs)
    assert eval_prob["eval_prob"] == pytest.approx(1.0)

    eval_prob = eval_probability(incorrect_probs)
    assert eval_prob["eval_prob"] == pytest.approx(0.0)
