import math

import pytest
import torch
from torch.utils.data import DataLoader

from arcsf.data.data_module import EvalQADataset, get_data, qa_formatter_autoregression
from arcsf.eval.metrics import (
    conditional_probability,
    eval_accuracy,
    ks_test,
    truth_ratio,
)
from arcsf.eval.utils import get_loss


@pytest.fixture
def dummy_data():
    return get_data(
        "tofu",
        granularity="author",
        stratified=True,
        forget_random=False,
        forgotten_author_fraction=1 / 3,
        forgotten_fact_fraction=1 / 3,
        random_seed=42,
    )


def test_accuracy():
    """Tests that accuracy function is outputting correct values"""
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
    """Tests that conditional probability gives the correct probabilities given a couple
    of edge cases.
    """
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
    """Tests that KS score is minimised given equal cdf functions"""
    probability_density, bin_edges = torch.histogram(
        torch.randn(1000), bins=10, density=True
    )
    bin_sizes = bin_edges[1:] - bin_edges[0:-1]
    test_cdf = torch.cumsum(probability_density * bin_sizes, dim=-1)
    comparison = ks_test(test_cdf, test_cdf)
    assert comparison[0] == pytest.approx(0.0)


def test_loss(dummy_tokenizer, dummy_forget_model):
    """Tests that the loss works as intended, outputting a loss with a gradient

    Args:
        dummy_tokenizer : tokenizer
        dummy_forget_model model
    """
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
    """Tests to see if the truth ratio works as intended, given some edge cases"""
    n_perturbed = 5
    correct_losses = torch.full((4, (n_perturbed + 1)), 10000)
    correct_losses[:, 0] = 0
    incorrect_losses = torch.full_like(correct_losses, 10000)
    incorrect_losses[:, 1:] = 0
    correct_ratio = truth_ratio(correct_losses)
    incorrect_ratio = truth_ratio(incorrect_losses)

    assert correct_ratio[0] == pytest.approx(0)
    assert math.isinf(incorrect_ratio[0])


# end-to-end test
def test_eval_end_to_end(dummy_base_model, dummy_tokenizer, dummy_data):
    """End-to-end test to ensure the evaluation pipeline works as intended.

    Args:
        dummy_base_model : model for tests
        dummy_tokenizer : tokenizer for the dummy model
        dummy_data : dummy data for tests
    """
    batch_size = 3
    n_perturbed = 1

    _, retain_data = dummy_data

    eval_dataset = EvalQADataset(
        data=retain_data,
        tokenizer=dummy_tokenizer,
        qa_formatter=qa_formatter_autoregression,
        loss_type="standard",
        return_perturbed=True,
        n_perturbed=n_perturbed,
    )
    dataloader = DataLoader(eval_dataset, batch_size=batch_size)

    formatted_inputs = next(iter(dataloader))

    all_losses = torch.zeros((batch_size, n_perturbed + 1))

    gt_batch = formatted_inputs[0]

    gt_dummy_model_output = dummy_base_model(
        input_ids=gt_batch["input_ids"],
        labels=gt_batch["labels"],
        attention_mask=gt_batch["attention_mask"],
    )
    gt_loss = get_loss(gt_dummy_model_output.logits, gt_batch["labels"])

    all_losses[:, 0] = gt_loss
    for perturbed_index in range(1, n_perturbed + 1):
        p_batch = formatted_inputs[perturbed_index]
        p_dummy_model_output = dummy_base_model(**p_batch)
        all_losses[:, perturbed_index] = get_loss(
            p_dummy_model_output.logits, p_batch["labels"]
        )

    means = torch.mean(all_losses, dim=0)
    tr = truth_ratio(all_losses)
    # checks the model performs better on the ground truth
    assert torch.all(tr < 1).item()
    # checks truth ratio is less than 1
    assert (means[0] < torch.min(means[1:])).item()
