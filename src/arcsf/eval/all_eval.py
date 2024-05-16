import argparse
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from arcsf.data.data_module import EvalQADataset, get_data, qa_formatter_autoregression
from arcsf.eval.metrics import eval_rouge_recall, truth_ratio
from arcsf.eval.utils import get_loss
from arcsf.utils import get_device


def all_eval(
    model: torch.nn.Module,
    dataset: EvalQADataset,
    batch_size: int,
    device: torch.device,
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
        tr : truth_ratio metric from the TOFU paper
        all_losses : input-wise losses for each ground truth and perturbed input
    """
    # move model to device create dataloader, initialise all_losses tensor
    model = model.to(device)
    n_perturbed = dataset.n_perturbed
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    output_dict = {
        "all_losses": torch.zeros((dataset.__len__(), n_perturbed + 1)),
        "truth_ratios": torch.zeros(dataset.__len__()),
        "rouge1_recall": torch.zeros(dataset.__len__()),
        "rougeL_recall": torch.zeros(dataset.__len__()),
    }
    # loop over batches
    for batch_idx, batch in enumerate(tqdm(data_loader, desc="Batch")):
        logit_batch, (question, answer) = batch
        gt_batch = logit_batch[0]
        batch_start_index = batch_idx * batch_size
        batch_end_index = batch_idx * batch_size + len(gt_batch["input_ids"])
        # don't need gradient
        with torch.no_grad():
            gt_outputs = model(**gt_batch)
            target_text = tokenizer.decode(answer["input_ids"][0][0])
            gen_outputs = model.generate(
                question["input_ids"][0],
                attention_mask=question["attention_mask"][0],
                **generate_kwargs,
            )
            generated_text = tokenizer.decode(
                gen_outputs[0][len(question["input_ids"][0][0]) :],
                skip_special_tokens=True,
            )
            for rouge_val_key in ["rouge1_recall", "rougeL_recall"]:
                output_dict[rouge_val_key][batch_start_index:batch_end_index] = (
                    eval_rouge_recall(
                        gen_output=generated_text, ground_truth=target_text
                    )[rouge_val_key]
                )
        # get ground truth loss
        gt_loss = get_loss(gt_outputs.logits, gt_batch["labels"])
        output_dict["all_losses"][batch_start_index:batch_end_index, 0] = gt_loss
        # loop over perturbed samples to get their losses
        for perturbed_index in range(1, n_perturbed + 1):
            p_batch = logit_batch[perturbed_index]
            with torch.no_grad():
                p_output = model(**p_batch)
            output_dict["all_losses"][
                batch_start_index:batch_end_index, perturbed_index
            ] = get_loss(p_output.logits, p_batch["labels"])

    # calculate truth_ratio and return them along with losses
    means = torch.mean(output_dict["all_losses"], dim=0)
    print(f"Mean losses:\n\tground_truth:{means[0]}\n\tperturbed:{means[1:].numpy()}")

    output_dict["truth_ratios"] = truth_ratio(output_dict["all_losses"])

    return output_dict


if __name__ == "__main__":
    # currently this is our random seed
    rand = 42

    parser = argparse.ArgumentParser(
        description=(
            "Runs quantitative evaluation, comparing output logits of model against"
            " targets. It currently saves the truth ratio and input-wise losses."
        )
    )
    batch_size = 1
    parser.add_argument("directory", type=str, help="Relative path to model directory.")
    parser.add_argument(
        "--data_split",
        "-s",
        default="forget",
        type=str,
        help="Split of data to evaluate on",
    )
    args = parser.parse_args()
    model_dir = args.directory

    # these are hardcoded for now
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)

    # experiment_config

    model.config.pad_token_id = tokenizer.eos_token_id

    experiment_config = yaml.safe_load(open(model_dir + "/experiment_config.yaml"))
    forget_data, retain_data = get_data(
        "tofu", **experiment_config["data_config"], random_seed=rand
    )
    splits = {
        "retain": retain_data,
        "forget": forget_data,
    }

    device = get_device()
    print(f"Pytorch device: {device}")
    qa_formatter = qa_formatter_autoregression
    dataset = EvalQADataset(
        splits[args.data_split],
        tokenizer,
        qa_formatter,
        "standard",
        quantitative_eval=True,
        qualitative_eval=True,
        device=get_device(),
        n_perturbed=2,
        random_seed=rand,
        padding=False,
    )

    # pass all to the function
    outputs = all_eval(
        model,
        dataset,
        batch_size,
        device,
        max_new_tokens=50,
    )
    save_dir = f"{model_dir}/eval/{args.data_split}/"
    os.makedirs(save_dir, exist_ok=True)
    np.savetxt(save_dir + "/truth_ratios.txt", outputs["truth_ratios"])
    np.savetxt(save_dir + "/all_losses.txt", outputs["all_losses"])
    np.savetxt(save_dir + "/rouge1_scores.txt", outputs["rouge1_recall"])
    np.savetxt(save_dir + "/rougeL_scores.txt", outputs["rougeL_recall"])
