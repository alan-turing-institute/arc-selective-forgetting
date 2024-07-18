import argparse
import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from scipy.stats import hmean
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from arcsf.data.data_module import (
    BlankQAFormatter,
    EvalQADataset,
    EvaluateDataCollator,
    QAFormatter,
    get_data,
)
from arcsf.eval.metrics import eval_rouge_recall, get_loss, ks_test, truth_ratio
from arcsf.eval.plot import plot_cdf
from arcsf.eval.utils import extract_qa_for_generate


def evaluate(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    forget_dataset: EvalQADataset,
    retain_dataset: EvalQADataset,
    base_truth_ratios: torch.Tensor | None,
    batch_size: int,
    n_print: int = 5,
    accelerator: Accelerator = Accelerator(),
    **generate_kwargs,
):
    print("Evaluating on forget data...")
    forget_metrics = compute_dataset_metrics(
        model,
        forget_dataset,
        batch_size,
        tokenizer,
        n_print=n_print,
        accelerator=accelerator,
        **generate_kwargs,
    )

    print("Evaluating on retain data...")
    retain_metrics = compute_dataset_metrics(
        model,
        retain_dataset,
        batch_size,
        tokenizer,
        n_print=n_print,
        accelerator=accelerator,
        **generate_kwargs,
    )

    print("Calculating forget quality and model utility...")
    return forget_quality_model_utility(
        base_truth_ratios, forget_metrics, retain_metrics
    )


def get_evaluate_inputs(
    forget_data: Dataset,
    retain_data: Dataset,
    qa_formatter: QAFormatter,
    loss_type: str,
    n_perturbed: int,
    random_seed: int,
    base_truth_ratios_path: str | None,
) -> dict[str, EvalQADataset | torch.Tensor]:
    # create eval datasets
    forget_dataset = EvalQADataset(
        forget_data,
        tokenizer,
        qa_formatter,
        loss_type,
        n_perturbed=n_perturbed,
        random_seed=random_seed,
    )
    retain_dataset = EvalQADataset(
        retain_data,
        tokenizer,
        qa_formatter,
        loss_type,
        n_perturbed=n_perturbed,
        random_seed=random_seed,
    )
    base_truth_ratios = EvaluateOutputs.load(base_truth_ratios_path).forget_truth_ratios
    return {
        "forget": forget_dataset,
        "retain": retain_dataset,
        "base_truth_ratios": base_truth_ratios,
    }


def compute_dataset_metrics(
    model: PreTrainedModel,
    dataset: EvalQADataset,
    batch_size: int,
    tokenizer: PreTrainedTokenizer,
    n_print: int = 0,
    accelerator: Accelerator = Accelerator(),
    **generate_kwargs: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes losses, truth ratios, and rouge scores for a model on a dataset.

    Args:
        model : Transformers model used to perform evaluation on
        dataset : Dataset to perform evaluation on
        batch_size : batch size for dataloader
        device : Pytorch device on which to perform computation
        n_print : number of samples to print the generated answers for
        accelerator : Accelerator object to handle moving data etc. to device

    Returns:
        output_dict : output dictionary with the values for analysis
    """
    n_perturbed = dataset.n_perturbed
    tokenizer.padding_side = "left"
    eval_collate_fn = EvaluateDataCollator(tokenizer=tokenizer)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=eval_collate_fn,
    )

    # Make data loader that returns batches of separated questions and ground truth
    # answers (so this can then be passed to accelerate to handle moving data to device)
    # batch[0] below selects the ground truth q/a pair (rather than perturbed answers)
    # batch size 1 as batch already created in previous data loader
    qa = [extract_qa_for_generate(batch[0], tokenizer) for batch in data_loader]
    qa_loader = DataLoader(qa, batch_size=1, collate_fn=lambda x: x[0], shuffle=False)

    # handle moving model and data loaders to device
    model, data_loader, qa_loader = accelerator.prepare(model, data_loader, qa_loader)

    dataset_len = len(dataset)
    output_dict = {
        "all_losses": torch.zeros((dataset_len, n_perturbed + 1), dtype=torch.float64),
        "truth_ratios": torch.zeros(dataset_len),
        "rougeL_recall": torch.zeros(dataset_len),
        "rouge1_recall": torch.zeros(dataset_len),
    }

    # =========
    # Metrics on logits of ground truth and perturbed answers
    # =========
    for batch_idx, batch in enumerate(
        tqdm(data_loader, desc="GT and Perturbed Logits")
    ):
        gt_batch = batch[0]
        pt_batches = batch[1:]
        batch_start_index = batch_idx * batch_size
        batch_end_index = batch_start_index + batch_size

        # Ground truth answers
        with torch.no_grad():
            # DO pass position_ids into this method -> see collator for info
            gt_outputs = model(**gt_batch)

        gt_loss = get_loss(gt_outputs.logits, gt_batch["labels"])
        output_dict["all_losses"][batch_start_index:batch_end_index, 0] = gt_loss.cpu()

        # Perturbed answers
        for perturbed_index in range(n_perturbed):
            pt_batch = pt_batches[perturbed_index]
            with torch.no_grad():
                # DO pass position_ids into this method -> see collator for info
                p_output = model(**pt_batch)
            output_dict["all_losses"][
                batch_start_index:batch_end_index, perturbed_index + 1
            ] = get_loss(p_output.logits, pt_batch["labels"]).cpu()

    # =========
    # Metrics on generated answers vs. actual ground truth answers
    # =========
    for batch_idx, qa in enumerate(tqdm(qa_loader, desc="Generate")):
        batch_start_index = batch_idx * batch_size
        questions, answers = qa

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

        if batch_idx * batch_size < n_print:
            n_print_batch = min(batch_size, n_print - batch_idx * batch_size)
            question_text = tokenizer.batch_decode(
                questions["input_ids"][:n_print_batch]
            )
            for q_text, gen_text, target_text in zip(
                question_text,
                generated_answers[:n_print_batch],
                target_answers[:n_print_batch],
            ):
                print(f"\nQuestion: {q_text}\n")
                print(f"Target: {target_text}\n")
                print(f"Generated: {gen_text}")
                print("-" * 15)

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

    # calculate truth_ratio and return them along with losses
    output_dict["truth_ratios"] = truth_ratio(output_dict["all_losses"])

    return output_dict


def forget_quality_model_utility(
    base_truth_ratios: torch.Tensor | None,
    forget_metrics: dict[str, torch.Tensor],
    retain_metrics: dict[str, torch.Tensor],
) -> "EvaluateOutputs":
    """
    Compute the forget quality and model utility metrics for a model.

    Args:
        base_truth_ratios: base model truth ratios to compare against in the ks_test. If
            None, the forget_quality metrics will not be calculated.
        forget_metrics: metrics from compute_dataset_metrics for the forget data
        retain_metrics: metrics from compute_dataset_metrics for the retain data

    Returns:
        EvaluateOutputs instance contatining the input forget and retain metrics
        (with keys forget_* and retain_*) as well as the following metrics:
            forget_quality_1 : 1-tailed ks-test comparison between forget and base model
                truth ratios
            forget_quality_2 : 2-tailed ks-test comparison between forget and base model
                truth ratios
            retain_mean_tr : mean of the retain truth ratios
            retain_mean_rougeL_recall : mean of the rougeL recall on the retain data
            retain_model_utility : harmonic mean of the retain mean truth ratio and mean
                rougeL recall
    """
    results_dict = {}

    # --------------
    # Metrics on forget dataset (i.e. quality of forgetting)
    # --------------
    if base_truth_ratios is not None:
        # one-tailed: truth ratio CDF for forget is greater than the base model
        results_dict["forget_quality_1"] = np.log(
            ks_test(
                forget_metrics["truth_ratios"], base_truth_ratios, alternative="greater"
            )
        ).item()
        # two-tailed: truth ratio CDF for forget is close to base model
        results_dict["forget_quality_2"] = np.log(
            ks_test(forget_metrics["truth_ratios"], base_truth_ratios)
        ).item()

    # --------------
    # Metrics on retain dataset (i.e. performance of model after forgetting)
    # --------------
    # need to transform truth ratios according to table 1 in the TOFU paper
    transform_retain_tr = torch.clamp((1 - retain_metrics["truth_ratios"]), 0)
    results_dict["retain_mean_tr"] = torch.mean(transform_retain_tr).item()

    results_dict["retain_mean_rougeL_recall"] = torch.mean(
        torch.tensor(retain_metrics["rougeL_recall"])
    ).item()

    results_dict["retain_model_utility"] = hmean(
        [results_dict["retain_mean_tr"], results_dict["retain_mean_rougeL_recall"]]
    )

    return EvaluateOutputs.combine_outputs(forget_metrics, retain_metrics, results_dict)


@dataclass
class EvaluateOutputs:
    forget_quality_1: float | None
    forget_quality_2: float | None
    forget_all_losses: torch.Tensor
    forget_truth_ratios: torch.Tensor
    forget_rougeL_recall: torch.Tensor
    forget_rouge1_recall: torch.Tensor
    retain_mean_tr: float
    retain_mean_rougeL_recall: float
    retain_model_utility: float
    retain_all_losses: torch.Tensor
    retain_truth_ratios: torch.Tensor
    retain_rougeL_recall: torch.Tensor
    retain_rouge1_recall: torch.Tensor

    def save(self, path: str) -> None:
        """
        Save an EvaluateOutputs instance to a json file.
        """
        with open(path, "w") as f:
            json.dump(self.to_raw_dict(), f)

    def to_raw_dict(self) -> dict[str, float | list | None]:
        """
        Converts the EvaluateOutputs instance to a dictionary, converting any torch
        tensors or numpy arrays to lists for compatibility with saving as JSON/logging
        to WandB.
        """
        out_dict = asdict(self)
        for key in out_dict:
            if isinstance(out_dict[key], np.ndarray):
                out_dict[key] = out_dict[key].tolist()
            if isinstance(out_dict[key], torch.Tensor):
                out_dict[key] = out_dict[key].cpu().numpy().tolist()
        return out_dict

    @classmethod
    def combine_outputs(
        cls,
        forget_metrics: dict[str, torch.Tensor],
        retain_metrics: dict[str, torch.Tensor],
        results_dict: dict[str, float | None],
    ) -> "EvaluateOutputs":
        """
        Creates an EvaluateOutputs instance from separate forget metrics, retain
        metrics, and forget quality/model utility as computed by
        forget_quality_model_utility.
        """
        # also raw forget and retain metrics to results dict with releveant prefix
        for prefix, metrics_dict in [
            ("forget", forget_metrics),
            ("retain", retain_metrics),
        ]:
            for key, values in metrics_dict.items():
                results_dict[f"{prefix}_{key}"] = values

        return cls.from_dict(results_dict)

    @classmethod
    def from_dict(cls, data: dict) -> "EvaluateOutputs":
        """
        Creates a EvaluateOutputs instance from a dictionary of data, converting any
        lists to torch tensors.
        """
        for key in data:
            if isinstance(data[key], list):
                data[key] = torch.tensor(data[key])
        return cls(**data)

    @classmethod
    def load(cls, path: str) -> "EvaluateOutputs":
        """
        Load an EvaluateOutputs instance from a json file.
        """
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    def __str__(self) -> str:
        return (
            f"FORGET SPLIT:\n"
            f"- One-Sided Forget Quality: {self.forget_quality_1}\n"
            f"- Two-Sided Forget Quality: {self.forget_quality_2}\n"
            f"RETAIN SPLIT:\n"
            f"- Model Utility: {self.retain_model_utility}\n"
            f"- Mean Truth Ratio: {self.retain_mean_tr}\n"
            f"- Mean Rouge: {self.retain_mean_rougeL_recall}\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Runs qualitative evaluation, comparing outputs of model"
            " against target strings."
        )
    )
    parser.add_argument(
        "model_path", type=str, help="Relative path to model directory."
    )
    parser.add_argument(
        "base_vals_path",
        type=str,
        help="Relative path to base model truth ratios.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "-b",
        "--eval_batch_size",
        type=int,
        default=16,
        help="Batch size for the evaluation pipeline to use. Defaults to 16.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Whether to plot the CDF of the model's performance.",
    )
    args = parser.parse_args()

    model_dir = args.model_path
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.config.pad_token_id = tokenizer.eos_token_id
    exp_config = yaml.safe_load(open(f"{model_dir}/experiment_config.yaml"))

    random_seed = exp_config["seed"]

    # get splits
    forget_data, retain_data = get_data(
        exp_config["data_config"]["dataset_name"],
        **exp_config["data_config"]["data_kwargs"],
        random_seed=random_seed,
    )

    eval_inputs = get_evaluate_inputs(
        forget_data,
        retain_data,
        BlankQAFormatter(),
        "standard",
        2,
        random_seed,
        args.base_vals_path,
    )

    eval_results = evaluate(
        model,
        tokenizer,
        eval_inputs["forget"],
        eval_inputs["retain"],
        eval_inputs["base_truth_ratios"],
        args.eval_batch_size,
        max_new_tokens=50,
    )

    save_dir = f"{model_dir}/eval/analysis/"
    os.makedirs(save_dir, exist_ok=True)
    eval_results.save(f"{save_dir}/metrics.json")

    exp_name = exp_config["experiment_name"]
    print(f"\nBase Model path: {args.base_vals_path}")
    print(f"Test Model path: {model_dir}")
    print(f"Experiment Name: {exp_name}")
    print(eval_results)

    if args.plot:
        plot_cdf(
            eval_inputs["base_truth_ratios"],
            eval_results.forget_truth_ratios,
            model_dir,
            exp_name,
            "forget",
        )
