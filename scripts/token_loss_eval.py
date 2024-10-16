import argparse
import json
import os

import torch
import yaml
from torch.nn.functional import softmax
from tqdm import tqdm

from arcsf.config.experiment import (
    DATA_CONFIG_DIR,
    EXPERIMENT_CONFIG_DIR,
    ExperimentConfig,
)
from arcsf.data.data_module import QAFormatter, get_data, get_idk_responses
from arcsf.eval.evaluate import EvaluateOutputs, Evaluator
from arcsf.eval.metrics import loss_function
from arcsf.models.model import load_model_and_tokenizer
from arcsf.utils import get_device, get_model_path


class IDKBatchGen:
    def __init__(self, random_seed, tokenizer, device) -> None:
        self.responses = get_idk_responses()
        self.rand_gen = torch.Generator().manual_seed(random_seed)
        self.tokenizer = tokenizer
        self.device = device
        self.pad_values = {
            "input_ids": self.tokenizer.pad_token_id,
            "attention_mask": 0,
            "position_ids": 0,
            "labels": -100,
        }

    def __call__(self, question_batch):
        """returns randomly sampled "I don't know" answer"""
        batch = {
            key: [None] * len(question_batch["input_ids"])
            for key in question_batch.keys()
        }
        for q_index, question in enumerate(question_batch["input_ids"]):
            rand_pos = torch.randint(
                0, len(self.responses), (1,), generator=self.rand_gen
            ).item()
            a = tokenizer.encode(self.responses[rand_pos])
            q = question.detach().tolist()
            batch["input_ids"][q_index] = q + a
            batch["attention_mask"][q_index] = question_batch["attention_mask"][
                q_index
            ].detach().tolist() + [1] * len(a)
            # start cumulative sum at ith index to ensure the first target token is 0
            batch["position_ids"][q_index] = [
                sum(batch["attention_mask"][q_index][:i])
                for i in range(len(batch["attention_mask"][q_index]))
            ]
            batch["labels"][q_index] = [-100] * len(q) + a

        max_len = max([len(input_ids) for input_ids in batch["input_ids"]])

        for key, values in batch.items():
            pad_value = self.pad_values[key]
            padded = [None] * len(values)
            for val_index, item in enumerate(values):
                pad_length = max_len - len(item)
                padded_item = [pad_value] * pad_length + item
                padded[val_index] = torch.tensor(padded_item).to(self.device)
            batch[key] = padded

        loaded_batch = {key: torch.stack(values) for key, values in batch.items()}

        return loaded_batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Token-based loss for a given model.")
    )
    parser.add_argument("--experiment_name", type=str, help="Relative path to model.")
    parser.add_argument("--save_retain", action="store_true")
    parser.add_argument("--full_model", action="store_true")
    args = parser.parse_args()

    experiment_path = args.experiment_name
    experiment_config = ExperimentConfig.from_yaml(
        EXPERIMENT_CONFIG_DIR / f"{experiment_path}.yaml"
    )
    data_config = yaml.safe_load(
        open(f"{DATA_CONFIG_DIR}/{experiment_config.config_names['data_config']}.yaml")
    )

    experiment_name = experiment_config.experiment_name

    if args.full_model:
        train_type = "full"
    else:
        train_type = experiment_config.train_type

    # getting the experiment config
    retain_model_dir = get_model_path(experiment_name, "retain")
    exp_config = yaml.safe_load(open(f"{retain_model_dir}/experiment_config.yaml"))

    if train_type == "full":
        target_model_dir = get_model_path(exp_config["full_model_name"], "full")
    else:
        target_model_dir = get_model_path(experiment_name, train_type)

    print(f"Model loaded at: {target_model_dir}")

    # load model from full model directory
    model, tokenizer = load_model_and_tokenizer(
        model_id=target_model_dir,
        peft_kwargs=exp_config["model_config"]["peft_kwargs"],
        **exp_config["model_config"]["model_kwargs"],
        add_padding_token=exp_config["model_config"]["add_padding_token"],
    )
    model.eval()

    # load experiment config from the retain model

    qa_formatter = QAFormatter(**exp_config["model_config"]["qa_formatter_kwargs"])
    n_perturbed = 2
    random_seed = exp_config["seed"]

    # get splits
    forget_split, retain_split = get_data(
        data_config["dataset_name"],
        **data_config["data_kwargs"],
        random_seed=random_seed,
    )

    compare_eval = EvaluateOutputs.load(f"{retain_model_dir}/eval_outputs.json")

    b_sz = exp_config["model_config"]["trainer_kwargs"]["per_device_eval_batch_size"]
    evaluator = Evaluator(
        model,
        forget_split,
        retain_split,
        qa_formatter,
        exp_config["data_config"]["dataset_name"],
        tokenizer,
        n_perturbed,
        random_seed,
        compare_eval.forget_truth_ratios,
        b_sz,
        max_new_tokens="adaptive",
    )

    forget_loader = evaluator.forget_loaders["full"]
    retain_loader = evaluator.retain_loaders["full"]

    model, forget_dataloader, retain_dataloader = evaluator.accelerator.prepare(
        model, forget_loader, retain_loader
    )

    if args.save_retain:
        loop_zip = (["forget", "retain"], [forget_dataloader, retain_dataloader])
    else:
        loop_zip = (["forget"], [forget_dataloader])

    idk_gen = IDKBatchGen(random_seed, tokenizer, device=get_device())
    for split, data_loader in zip(*loop_zip):
        token_loss_dict = {
            "target_losses": [],
            "target_labels": [],
            "target_probs": [],
            "idk_losses": [],
            "idk_labels": [],
            "idk_probs": [],
        }
        for batch in tqdm(data_loader, desc=f"{split} batch"):
            ground_truth_batch = batch[0]
            idk_batch = idk_gen(batch[1])

            with torch.no_grad():
                gt_output = model(**ground_truth_batch)
                idk_output = model(**idk_batch)
            # logits
            gt_output_logits = gt_output.logits[..., :-1, :].contiguous()
            idk_output_logits = idk_output.logits[..., :-1, :].contiguous()
            # labels
            gt_shifted_labels = ground_truth_batch["labels"][..., 1:].contiguous()
            idk_shifted_labels = idk_batch["labels"][..., 1:].contiguous()
            # probabilities
            gt_probabilities = softmax(gt_output_logits, dim=-1)
            idk_probabilities = softmax(idk_output_logits, dim=-1)

            batch_size, gt_sequence_length, _ = gt_probabilities.shape
            _, idk_sequence_length, _ = idk_probabilities.shape

            gt_loss = loss_function(
                gt_output_logits.transpose(-1, -2), gt_shifted_labels
            )
            idk_loss = loss_function(
                idk_output_logits.transpose(-1, -2), idk_shifted_labels
            )

            for sample_n in range(batch_size):
                # gt
                gt_probs = gt_probabilities[sample_n, ...]
                sample_target_probs = gt_probs[
                    range(gt_sequence_length), gt_shifted_labels[sample_n, :]
                ]
                gt_target_idxs = gt_shifted_labels[sample_n, :] != -100
                # idk
                idk_probs = idk_probabilities[sample_n, ...]
                sample_idk_probs = idk_probs[
                    range(idk_sequence_length), idk_shifted_labels[sample_n, :]
                ]
                idk_target_idxs = idk_shifted_labels[sample_n, :] != -100

                token_loss_dict["target_losses"].append(
                    gt_loss[sample_n, gt_target_idxs].detach().cpu().tolist()
                )
                token_loss_dict["target_labels"].append(
                    gt_shifted_labels[sample_n, gt_target_idxs].detach().cpu().tolist()
                )
                token_loss_dict["target_probs"].append(
                    sample_target_probs[gt_target_idxs].detach().cpu().tolist()
                )

                token_loss_dict["idk_losses"].append(
                    idk_loss[sample_n, idk_target_idxs].detach().cpu().tolist()
                )
                token_loss_dict["idk_labels"].append(
                    idk_shifted_labels[sample_n, idk_target_idxs]
                    .detach()
                    .cpu()
                    .tolist()
                )
                token_loss_dict["idk_probs"].append(
                    sample_idk_probs[idk_target_idxs].detach().cpu().tolist()
                )

        if train_type == "full":
            target_model_dir = (
                f"{target_model_dir}/eval_outputs/"
                f"{exp_config['config_names']['data_config']}"
            )
            os.makedirs(target_model_dir, exist_ok=True)

        save_path = f"{target_model_dir}/{split}_token_loss.json"
        with open(save_path, "w") as f:
            json.dump(token_loss_dict, f)
