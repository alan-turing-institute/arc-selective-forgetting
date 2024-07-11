import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from scipy.stats import hmean
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorWithPadding, PreTrainedTokenizer

from arcsf.data.data_module import EvalQADataset, EvaluateDataCollator
from arcsf.eval.metrics import eval_rouge_recall, ks_test, truth_ratio

_loss_function = CrossEntropyLoss(ignore_index=-100, reduction="none")


def check_nans(array: np.ndarray | torch.Tensor, name: str = "") -> None:
    if name != "":
        name = " " + name
    message = f"NaNs found in{name} array of shape: {array.shape}"
    if isinstance(array, np.ndarray):
        if np.isnan(array).any():
            warnings.warn(message)
    elif isinstance(array, torch.Tensor):
        if torch.isnan(array).any():
            warnings.warn(message)


def get_loss(output_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute loss along a batch from the evaluation script

    Args:
        output_logits: output logits from model
        (batch_size x sequence_length x vocab_size)
        labels: labels (batch_size x sequence_length)

    Returns:
        _description_
    """
    # shape: batch_size x (sequence_length-1) x vocab_size
    output_logits = output_logits[..., :-1, :].contiguous()

    # shape : batch_size x (sequence_length - 1)
    shifted_labels = labels[..., 1:].contiguous()
    # output_logits.transpose(-1, -2) shape: batch_size x vocab x (sequence_length - 1)
    # loss shape: batch_size
    loss = _loss_function(output_logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
    target_len = torch.sum(labels != -100, dim=-1)  # length of tokens in target
    loss_normalised = loss / target_len  # normalised loss shape: batch_size
    return loss_normalised


def get_losses(output, targets):
    losses = torch.zeros(len(targets))
    for index, target in enumerate(targets):
        losses[index] = get_loss(output, target)
    return losses


def ecdf(x):
    xs, _ = torch.sort(x)
    ys = torch.arange(1, len(xs) + 1) / float(len(xs))
    return xs, ys


def combine_dicts(
    forget_dict: dict[str, np.ndarray | torch.Tensor],
    retain_dict: dict[str, np.ndarray | torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Combines the evaluation dictionaries from both models to return only the values used
    for the aggregation.

    Args:
        forget_dict: Evaluation metrics from the forget model
        retain_dict: Evaluation metrics from the retain model

    Returns:
        Dictionary containing the retain truth ratios, forget truth ratios, and forget
        ROUGE scores.
    """
    return {
        "retain_tr": retain_dict["truth_ratios"],
        "forget_tr": forget_dict["truth_ratios"],
        "rouge_scores": retain_dict["rougeL_recall"],
    }


def get_metrics(
    base_truth_ratios: torch.Tensor,
    test_values: dict[torch.Tensor],
) -> dict[str, float]:
    """
    Retrieves metrics for tracking an evaluation.

    Args:
        base_truth_ratios: base model truth ratios to compare against in the ks_test
        test_values: recorded values from the evaluation of the test model, including
        both ks test settings

    Returns:
        result_dict : dictionary contatining the metrics we would like to track
    """
    forget_quality_one_sided = np.log(
        # the case when the truth ratio CDF for forget is greater than the base model
        ks_test(test_values["forget_tr"], base_truth_ratios, alternative="greater")
    )
    forget_quality_two_sided = np.log(
        # the general measure of 'closeness' of the truth ratio CDF for forget
        # against the base model
        ks_test(test_values["forget_tr"], base_truth_ratios)
    )

    # need to transform these according to table 1 in the TOFU paper
    transform_retain_tr = torch.clamp((1 - test_values["retain_tr"]), 0)
    retain_tr = torch.mean(transform_retain_tr)
    # calculate other scores
    rouge_score = torch.mean(torch.tensor(test_values["rouge_scores"]))
    model_utilty = hmean([retain_tr, rouge_score])

    result_dict = {
        "mean_tr_retain": retain_tr.item(),
        "mean_rouge_score": rouge_score.item(),
        "forget_quality_1": forget_quality_one_sided.item(),
        "forget_quality_2": forget_quality_two_sided.item(),
        "model_utility": model_utilty.item(),
    }

    return result_dict


def first_idx(
    array: torch.tensor, value: int | float, equal: bool, dim: int
) -> torch.tensor:
    """
    If equal is True returns the indices of the first occurrence of a value
    in a tensor along a given dimension, or 0 if the value is not found.
    If equal is False returns the indices of the first occurrence of *not* the value,
    or 0 if the whole row of the tensor is equal to the value.

    Args:
        array : tensor to search
        value : value to search for
        equal: whether to search for the value, or anything not equal to the value
        dim : dimension to search along

    Returns:
        tensor of indices of the first occurrence of the value
    """
    if equal:
        bools = array == value
    else:
        bools = array != value
    return torch.argmax(bools.to(int), dim=dim)


def extract_qa_for_generate(
    inputs: dict[str, torch.tensor],
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
) -> tuple[dict[str, torch.tensor], dict[str, torch.tensor]]:
    """
    Uses masked labels (generated in EvalQADataset) to extract the questions and answers
    from a batch of formatted & tokenized combined question and answers.

    Args:
        inputs : batch of inputs for the model (tokenized and formatted combined QA)
        tokenizer : tokenizer used to tokenize the inputs.
        device : device to move output data to

    Returns:
        batch of left padded question input_ids and attention_mask, and batch of left
        padded answer question input_ids and attention_mask
    """
    if tokenizer.padding_side != "left":
        raise ValueError(
            "tokenizer.padding_side must be left for this function to work as intended"
        )
    collator = DataCollatorWithPadding(tokenizer, padding=True)

    # index of first non-masked token in labels of each sequence should indicate the
    # start of the answer (question prompt is everything before that)
    answer_start_ids = first_idx(inputs["labels"], value=-100, equal=False, dim=-1)
    questions = collator(
        [
            {
                "input_ids": inputs["input_ids"][idx, :start],
                "attention_mask": inputs["attention_mask"][idx, :start],
            }
            for idx, start in enumerate(answer_start_ids)
        ]
    )

    # The original combined QA is left padded to make all the QA the same length.
    # After extracting the questions by themselves they then have more padding than
    # necessary. Remove it here.
    n_trim_pad = torch.min(
        first_idx(questions["attention_mask"], value=1, equal=True, dim=-1)
    )
    questions["input_ids"] = questions["input_ids"][:, n_trim_pad:]
    questions["attention_mask"] = questions["attention_mask"][:, n_trim_pad:]

    answers = collator(
        [
            {
                "input_ids": inputs["input_ids"][idx, start:],
                "attention_mask": inputs["attention_mask"][idx, start:],
            }
            for idx, start in enumerate(answer_start_ids)
        ]
    )

    return questions.to(device), answers.to(device)


def all_eval(
    model: transformers.PreTrainedModel,
    dataset: EvalQADataset,
    batch_size: int,
    device: torch.device,
    tokenizer: transformers.PreTrainedTokenizer,
    **generate_kwargs: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Performs quantitative evaluation of the selected model over selected data.
    Returns the input-wise losses on target answers for the ground truth and perturbed
    answers.

    Args:
        model : Transformers model used to perform evaluation on
        dataset : Dataset to perform evaluation on
        batch_size : batch size for dataloader
        device : Pytorch device on which to perform computation

    Returns:
        output_dict : output dictionary with the values for analysis
    """
    # move model to device create dataloader, initialise all_losses tensor
    model = model.to(device)
    n_perturbed = dataset.n_perturbed
    tokenizer.padding_side = "left"
    eval_collate_fn = EvaluateDataCollator(tokenizer=tokenizer)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=eval_collate_fn,
    )
    dataset_len = len(dataset)
    output_dict = {
        "all_losses": torch.zeros((dataset_len, n_perturbed + 1), dtype=torch.float64),
        "truth_ratios": torch.zeros(dataset_len),
        "rougeL_recall": torch.zeros(dataset_len),
        "rouge1_recall": torch.zeros(dataset_len),
    }
    # loop over batches
    for batch_idx, batch in enumerate(tqdm(data_loader, desc="Batch")):
        gt_batch = batch[0]
        pt_batches = batch[1:]
        batch_start_index = batch_idx * batch_size
        batch_end_index = batch_start_index + batch_size

        # =========
        # Ground truth: Loss based on logits vs. actual answers
        # =========
        with torch.no_grad():
            # DO pass position_ids into this method -> see collator for info
            gt_outputs = model(**gt_batch)

        gt_loss = get_loss(gt_outputs.logits, gt_batch["labels"].to(device))
        output_dict["all_losses"][batch_start_index:batch_end_index, 0] = gt_loss.cpu()

        # =========
        # Ground truth: Metrics on generated answers vs. actual answers
        # =========
        questions, answers = extract_qa_for_generate(gt_batch, tokenizer, device)

        with torch.no_grad():
            # DO NOT pass position_ids into this method -> see collator for info
            gen_outputs = model.generate(
                input_ids=questions["input_ids"],
                attention_mask=questions["attention_mask"],
                pad_token_id=tokenizer.pad_token_id,
                **generate_kwargs,
            )

        target_answers = tokenizer.batch_decode(
            answers["input_ids"], skip_special_tokens=True
        )
        generated_answers = [
            tokenizer.decode(
                gen_a[len(q) :],  # only want the tokens for the answers
                skip_special_tokens=True,
            )
            for q, gen_a in zip(questions["input_ids"], gen_outputs)
        ]

        for rouge_idx, (generated_text, target_text) in enumerate(
            zip(generated_answers, target_answers)
        ):
            rouge_result = eval_rouge_recall(
                gen_output=generated_text, ground_truth=target_text
            )

            output_dict["rougeL_recall"][batch_start_index + rouge_idx] = rouge_result[
                "rougeL_recall"
            ]
            output_dict["rouge1_recall"][batch_start_index + rouge_idx] = rouge_result[
                "rouge1_recall"
            ]

        # =========
        # Perturbed answers: Loss based on logits vs. perturbed (incorrect) answers
        # =========
        for perturbed_index in range(n_perturbed):
            pt_batch = pt_batches[perturbed_index]
            with torch.no_grad():
                p_output = model(**pt_batch)
            output_dict["all_losses"][
                batch_start_index:batch_end_index, perturbed_index + 1
            ] = get_loss(p_output.logits, pt_batch["labels"]).cpu()

    # calculate truth_ratio and return them along with losses
    output_dict["truth_ratios"] = truth_ratio(output_dict["all_losses"])

    return output_dict


def get_analysis_values(
    model_dir: str,
) -> dict[str, np.ndarray | torch.Tensor]:
    """
    Gets the values for analysis given a model directory.
    Args:
        model_dir : file path to where model is stored with evaluation output

    Returns:
        vals : dictionary containing the values to be used in analysis
    """
    vals = dict()
    # losses are of shape (n_samples x n_perturbed)
    vals["forget_losses"] = np.loadtxt(model_dir + "/eval/forget/all_losses.txt")
    vals["retain_losses"] = np.loadtxt(model_dir + "/eval/retain/all_losses.txt")
    vals["rouge_scores"] = np.loadtxt(model_dir + "/eval/retain/rougeL_scores.txt")
    # we re-calculate the truth ratio, since torch calculated many as NaNs
    # truth ratios are of shape (n_samples)
    vals["forget_tr"] = truth_ratio(torch.tensor(vals["forget_losses"]))
    vals["retain_tr"] = torch.clamp(
        1 - truth_ratio(torch.tensor(vals["retain_losses"])),
        min=0,
    )
    for key in vals:
        check_nans(vals[key], key)
    return vals


def plot_cdf(
    base_vals: dict,
    test_vals: dict,
    save_path: str,
    exp_name: str,
    split: str = "retain",
) -> None:
    """
    Function to plot and save CDF functions for model comparison

    Args:
        base_vals : values for the base model to be compared against
        test_vals : values for the test model to be compared
        save_path : relative path where the model is stored
        exp_name : the name of the experiment for naming purposes
        split (optional): the split to be used. Defaults to "retain".
    """
    base_data, base_y_values = ecdf(base_vals[f"{split}_tr"])
    test_data, test_y_values = ecdf(test_vals[f"{split}_tr"])

    p_val = ks_test(test_vals[f"{split}_tr"], base_vals[f"{split}_tr"])

    plt.title(f"{split} data CDF")
    plt.plot(base_data, base_y_values, label="base-model")
    plt.plot(test_data, test_y_values, label=f"forget-model {exp_name}")
    plt.annotate(f"Model Utility {np.log(p_val)}", (0.9, 0.1))
    plt.legend()
    plt.xlim(0, 1)
    plt.savefig(save_path + f"/eval/analysis/tr-cdf-{split}-{exp_name}-.pdf")
    plt.close()


def plot_scatter(dict: dict[float], **plot_kwargs) -> None:
    """
    Plot of the data from a given model/experiment

    Args:
        dict : dictionary containing the aggrevated evaluation metrics of the model.
        plot_kwargs : key word arguments for the `matplotlib.pylot.scatter` function.
    """
    plt.scatter(dict["model_utility"], dict["forget_quality_1"], **plot_kwargs)
