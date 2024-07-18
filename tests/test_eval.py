import math

import pytest
import torch
from torch.utils.data import DataLoader

from arcsf.data.data_module import (
    BlankQAFormatter,
    EvalQADataset,
    EvaluateDataCollator,
    get_data,
)
from arcsf.eval.evaluate_model import evaluate_model
from arcsf.eval.metrics import (
    conditional_probability,
    eval_accuracy,
    get_loss,
    ks_test,
    truth_ratio,
)


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


@pytest.fixture
def dummy_exp_config():
    return {
        "data_config": {
            "dataset_name": "tofu",
            "data_kwargs": {
                "forget_random": False,
                "forgotten_author_fraction": 1 / 3,
                "forgotten_fact_fraction": 1 / 3,
                "granularity": "author",
                "stratified": True,
            },
        },
        "seed": 42,
    }


def test_accuracy():
    """Tests that accuracy function is outputting correct values"""
    # random predictions for test outputs
    test_outputs_correct = torch.randn(100, 5)
    # argmax predictions for test targets
    test_targets = test_outputs_correct.argmax(dim=1)
    # flip random predictions for incorrect predictions
    test_outputs_incorrect = test_outputs_correct * -1

    test_outputs_middle = torch.concat(
        [test_outputs_correct[:50, :], test_outputs_incorrect[50:, :]]
    )

    correct = eval_accuracy(test_outputs_correct, test_targets)
    assert correct["eval_accuracy"] == 1.0

    incorrect = eval_accuracy(test_outputs_incorrect, test_targets)
    assert incorrect["eval_accuracy"] == 0.0

    incorrect = eval_accuracy(test_outputs_middle, test_targets)
    assert incorrect["eval_accuracy"] == 0.5


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
    assert torch.mean(eval_prob[0]).item() == pytest.approx(1.0)

    eval_prob = conditional_probability(incorrect_losses)
    assert torch.mean(eval_prob[0]).item() == pytest.approx(0.0)


def test_ks_test():
    """Tests that KS score is minimised given equal cdf functions"""
    probability_density_1, bin_edges = torch.histogram(
        torch.randn(1000), bins=10, density=True
    )
    probability_density_2, bin_edges = torch.histogram(
        torch.randn(1000) * 2 + 1, bins=10, density=True
    )
    bin_sizes = bin_edges[1:] - bin_edges[0:-1]
    test_cdf_1 = torch.cumsum(probability_density_1 * bin_sizes, dim=-1)
    test_cdf_2 = torch.cumsum(probability_density_2 * bin_sizes, dim=-1)
    same_comparison = ks_test(test_cdf_1, test_cdf_1)
    different_comparison = ks_test(test_cdf_1, test_cdf_2)
    assert same_comparison == 1.0
    assert different_comparison != 1.0


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
    with torch.no_grad():
        test_output = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    loss = get_loss(test_output.logits, labels)

    assert isinstance(loss, torch.Tensor)


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
        qa_formatter=BlankQAFormatter(),
        loss_type="standard",
        n_perturbed=n_perturbed,
    )
    dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        # right padding can be used since generation not performed
        collate_fn=EvaluateDataCollator(tokenizer=dummy_tokenizer, padding_side="left"),
    )

    formatted_inputs = next(iter(dataloader))

    all_losses = torch.zeros((batch_size, n_perturbed + 1))

    gt_batch = formatted_inputs[0]

    gt_dummy_model_output = dummy_base_model(**gt_batch)
    gt_loss = get_loss(gt_dummy_model_output.logits, gt_batch["labels"])

    all_losses[:, 0] = gt_loss
    for perturbed_index in range(1, n_perturbed + 1):
        p_batch = formatted_inputs[perturbed_index]
        p_dummy_model_output = dummy_base_model(**p_batch)
        all_losses[:, perturbed_index] = get_loss(
            p_dummy_model_output.logits, p_batch["labels"]
        )
    tr = truth_ratio(all_losses)
    # checks truth ratio is less than 1
    assert torch.all(tr < 1).item()


#
def test_evaluate_model(dummy_base_model, dummy_tokenizer, dummy_exp_config):
    # we load in some random numbers for the truth ratios
    dummy_truth_ratios = "tests/data/dummy_truth_ratios.txt"
    # these are the metrics the function should output
    metric_keys = [
        "mean_tr_retain",
        "mean_rouge_score",
        "forget_quality_1",
        "forget_quality_2",
        "model_utility",
    ]

    test_eval = evaluate_model(
        dummy_base_model,
        dummy_truth_ratios,
        dummy_tokenizer,
        dummy_exp_config,
        batch_size=3,
        max_new_tokens=10,
    )

    # check we get the correct outputs and that theyre all native float
    assert list(test_eval.keys()) == metric_keys
    for key in metric_keys:
        assert isinstance(test_eval[key], float)


def test_data_collator(dummy_base_model, dummy_tokenizer, dummy_data):
    """End-to-end test to ensure the data collator works as intended when interacting
    with the model.

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
        qa_formatter=BlankQAFormatter(),
        loss_type="standard",
        n_perturbed=n_perturbed,
    )
    ls_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        # left padding collator
        collate_fn=EvaluateDataCollator(tokenizer=dummy_tokenizer, padding_side="left"),
    )

    rs_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        # right padding collator
        collate_fn=EvaluateDataCollator(
            tokenizer=dummy_tokenizer, padding_side="right"
        ),
    )

    left_padded_inputs = next(iter(ls_dataloader))
    right_padded_inputs = next(iter(rs_dataloader))

    ls_gt_batch = left_padded_inputs[0]
    rs_gt_batch = right_padded_inputs[0]

    left_padded_output = dummy_base_model(
        **ls_gt_batch,
    )
    right_padded_output = dummy_base_model(
        **rs_gt_batch,
    )

    # use the attention mask to find the relevant token positions and check they are
    # close to one another
    assert torch.sum(
        left_padded_output.logits[ls_gt_batch["attention_mask"] != 0].detach()
        - right_padded_output.logits[rs_gt_batch["attention_mask"] != 0].detach()
    ).numpy() == pytest.approx(0, abs=1e-3)

    ls_loss = get_loss(left_padded_output.logits, ls_gt_batch["labels"])
    rs_loss = get_loss(right_padded_output.logits, rs_gt_batch["labels"])

    # ensure the losses are very close to one another as to not influence results
    assert torch.sum(ls_loss.detach() - rs_loss.detach()).numpy() == pytest.approx(
        0, abs=1e-5
    )
