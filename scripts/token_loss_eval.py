import argparse
import json
import os

import torch
import yaml
from torch.nn.functional import softmax
from tqdm import tqdm

from arcsf.config.experiment import EXPERIMENT_CONFIG_DIR, ExperimentConfig
from arcsf.data.data_module import QAFormatter, get_data, get_idk_responses
from arcsf.eval.evaluate import EvaluateOutputs, Evaluator
from arcsf.eval.metrics import loss_function
from arcsf.models.model import load_model_and_tokenizer
from arcsf.utils import get_model_path


class IDKBatchGen:
    def __init__(self, random_seed, tokenizer) -> None:
        self.responses = get_idk_responses()
        self.rand_gen = torch.Generator().manual_seed(random_seed)
        self.tokenizer = tokenizer

    def __call__(self, question_batch):
        """returns randomly sampled "I don't know" answer"""
        batch = {
            key: [None] * len(question_batch)
            for key in ["input_ids", "attention_mask", "labels"]
        }
        for q_index, question in enumerate(question_batch):
            rand_pos = torch.randint(
                0, len(self.responses), (1,), generator=self.rand_gen
            ).item()
            answer = self.responses[rand_pos]
            qa = question["input_ids"] + tokenizer.tokenize(answer)
            print(qa)

        print(batch)


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
        exp_config["data_config"]["dataset_name"],
        **exp_config["data_config"]["data_kwargs"],
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

    idk_gen = IDKBatchGen(random_seed, tokenizer)
    for split, data_loader in zip(*loop_zip):
        token_loss_dict = {
            "all_losses": [],
            "all_labels": [],
            "all_probs": [],
            "target_losses": [],
            "target_labels": [],
            "target_probs": [],
        }
        for batch in tqdm(data_loader, desc=f"{split} batch"):
            ground_truth_batch = batch[0]
            idk_batch = idk_gen(batch[1])
            print(idk_batch)
            exit()
            with torch.no_grad():
                output = model(**ground_truth_batch)
            output_logits = output.logits[..., :-1, :].contiguous()
            shifted_labels = ground_truth_batch["labels"][..., 1:].contiguous()
            all_probabilities = softmax(output_logits, dim=-1)

            batch_size, sequence_length, vocab_size = all_probabilities.shape

            loss = loss_function(output_logits.transpose(-1, -2), shifted_labels)

            for sample_n in range(batch_size):
                sample_probs = all_probabilities[sample_n, ...]
                sample_target_probs = sample_probs[
                    range(sequence_length), shifted_labels[sample_n, :]
                ]

                target_idxs = shifted_labels[sample_n, :] != -100

                token_loss_dict["all_losses"].append(
                    loss[sample_n, :].detach().cpu().tolist()
                )
                token_loss_dict["all_labels"].append(
                    shifted_labels[sample_n, :].detach().cpu().tolist()
                )
                token_loss_dict["all_probs"].append(
                    sample_target_probs.detach().cpu().tolist()
                )

                token_loss_dict["target_losses"].append(
                    loss[sample_n, target_idxs].detach().cpu().tolist()
                )

                token_loss_dict["target_losses"].append(
                    loss[sample_n, target_idxs].detach().cpu().tolist()
                )
                token_loss_dict["target_labels"].append(
                    shifted_labels[sample_n, target_idxs].detach().cpu().tolist()
                )
                token_loss_dict["target_probs"].append(
                    sample_target_probs[target_idxs].detach().cpu().tolist()
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
