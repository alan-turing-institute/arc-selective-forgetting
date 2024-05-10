import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from arcsf.data.data_module import EvalQADataset, get_data, qa_formatter_autoregression
from arcsf.eval.metrics import truth_ratio
from arcsf.eval.utils import get_loss
from arcsf.utils import get_device


def quantitative_eval(model, dataset, batch_size, random_seed, device):
    model = model.to(device)
    n_perturbed = dataset.n_perturbed
    gen = torch.Generator().manual_seed(random_seed)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, generator=gen
    )
    all_losses = torch.zeros((dataset.__len__(), n_perturbed + 1))
    for batch_idx, batch in enumerate(tqdm(data_loader, desc="Batch")):
        gt_batch = batch[0]
        batch_start_index = batch_idx * batch_size
        batch_end_index = batch_idx * batch_size + len(gt_batch["input_ids"])
        with torch.no_grad():
            gt_dummy_model_output = model(**gt_batch)
        gt_loss = get_loss(gt_dummy_model_output.logits, gt_batch["labels"])

        all_losses[batch_start_index:batch_end_index, 0] = gt_loss
        for perturbed_index in range(1, n_perturbed + 1):
            p_batch = batch[perturbed_index]
            with torch.no_grad():
                p_dummy_model_output = model(**p_batch)
            all_losses[batch_start_index:batch_end_index, perturbed_index] = get_loss(
                p_dummy_model_output.logits, p_batch["labels"]
            )

    means = torch.mean(all_losses, dim=0)
    print(f"Mean losses:\n\tground_truth:{means[0]}\n\tperturbed:{means[1:]}")
    tr = truth_ratio(all_losses)

    return tr, all_losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Runs qualitative evaluation, comparing outputs of model"
            " against target strings."
        )
    )
    batch_size = 5
    parser.add_argument("directory", type=str, help="Relative path to model directory.")
    args = parser.parse_args()
    model_dir = args.directory

    rand = 42
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    model.config.pad_token_id = tokenizer.eos_token_id
    forget_data, retain_data = get_data(
        "tofu", "author", True, True, 0.2, 0.2, random_seed=rand
    )
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
    truth_ratios, all_losses = quantitative_eval(
        model, dataset, batch_size, rand, device
    )
    np.savetxt(model_dir + "/../eval/truth_ratios.txt", truth_ratios)
    np.savetxt(model_dir + "/../eval/all_losses.txt", all_losses)
