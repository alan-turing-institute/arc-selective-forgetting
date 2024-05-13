import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from arcsf.data.data_module import EvalQADataset, get_data, qa_formatter_autoregression
from arcsf.eval.metrics import truth_ratio
from arcsf.eval.utils import get_loss
from arcsf.utils import get_device


def quantitative_eval(
    model: torch.nn.Module,
    dataset: Dataset,
    batch_size: int,
    device: torch.device,
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
        tr : truth_ratio metric from the TOFU paper
        all_losses : input-wise losses for each ground truth and perturbed input
    """
    # move model to device create dataloader, initialise all_losses tensor
    model = model.to(device)
    n_perturbed = dataset.n_perturbed
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_losses = torch.zeros((dataset.__len__(), n_perturbed + 1))
    # loop over batches
    for batch_idx, batch in enumerate(tqdm(data_loader, desc="Batch")):
        gt_batch = batch[0]
        batch_start_index = batch_idx * batch_size
        batch_end_index = batch_idx * batch_size + len(gt_batch["input_ids"])
        # don't need gradient
        with torch.no_grad():
            gt_dummy_model_output = model(**gt_batch)
        # get ground truth loss
        gt_loss = get_loss(gt_dummy_model_output.logits, gt_batch["labels"])
        all_losses[batch_start_index:batch_end_index, 0] = gt_loss
        # loop over perturbed samples to get their losses
        for perturbed_index in range(1, n_perturbed + 1):
            p_batch = batch[perturbed_index]
            with torch.no_grad():
                p_dummy_model_output = model(**p_batch)
            all_losses[batch_start_index:batch_end_index, perturbed_index] = get_loss(
                p_dummy_model_output.logits, p_batch["labels"]
            )
    # calculate truth_ratio and return them along with losses
    means = torch.mean(all_losses, dim=0)
    print(f"Mean losses:\n\tground_truth:{means[0]}\n\tperturbed:{means[1:]}")
    tr = truth_ratio(all_losses)

    return tr, all_losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Runs quantitative evaluation, comparing output logits of model against"
            " targets. It currently saves the truth ratio and input-wise losses."
        )
    )
    batch_size = 5
    parser.add_argument("directory", type=str, help="Relative path to model directory.")
    args = parser.parse_args()
    model_dir = args.directory

    # these are hardcoded for now
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model.config.pad_token_id = tokenizer.eos_token_id
    forget_data, retain_data = get_data("tofu", "author", True, True, 0.2, 0.2)
    device = get_device()
    print(f"Pytorch device: {device}")
    qa_formatter = qa_formatter_autoregression
    dataset = EvalQADataset(
        retain_data,
        tokenizer,
        qa_formatter,
        "standard",
        device=get_device(),
        n_perturbed=2,
    )
    # pass all to the function
    truth_ratios, all_losses = quantitative_eval(model, dataset, batch_size, device)
    np.savetxt(model_dir + "/../eval/truth_ratios.txt", truth_ratios)
    np.savetxt(model_dir + "/../eval/all_losses.txt", all_losses)
