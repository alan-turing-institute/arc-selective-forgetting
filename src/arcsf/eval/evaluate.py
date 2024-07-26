import json
import logging
from copy import deepcopy
from dataclasses import asdict, dataclass

import numpy as np
import torch
from accelerate import Accelerator
from datasets import Dataset
from scipy.stats import hmean
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from arcsf.data.data_module import EvalQADataset, EvaluateDataCollator, QAFormatter
from arcsf.eval.metrics import eval_rouge_recall, get_loss, ks_test, truth_ratio
from arcsf.eval.utils import extract_qa_for_generate

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(
        self,
        model: PreTrainedModel,
        forget_split: Dataset,
        retain_split: Dataset,
        qa_formatter: QAFormatter,
        dataset_name: str,
        tokenizer: PreTrainedTokenizer,
        n_perturbed: int,
        random_seed: int,
        base_truth_ratios: torch.Tensor | None,
        batch_size: int,
        accelerator: Accelerator = Accelerator(),
        n_print: int = 5,
        **generate_kwargs: dict,
    ):
        """
        Create an Evaluator instance, used to compute forget quality and model utility
        metrics for a model.

        Args:
            model : model to evaluate
            forget_split : forget data to evaluate on
            retain_split : retain data to evaluate on
            qa_formatter : QAFormatter instance to format questions and answers
            tokenizer : tokenizer to encode and decode text
            n_perturbed : number of perturbed answers to evaluate for each question
            random_seed : random seed for reproducibility
            base_truth_ratios : base model truth ratios to compare against in the
                when computing forget quality. If None, the forget quality metrics will
                not be calculated.
            batch_size : batch size for evaluation
        """
        # create a copy of the tokenizer (to avoid changing behaviour of original)
        # and switch it to use left padding
        tokenizer = deepcopy(tokenizer)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            if tokenizer.bos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.bos_token_id
            else:
                tokenizer.pad_token_id = tokenizer.eos_token_id

        logger.info("Creating forget loaders...")
        self.forget_loaders = self.get_eval_data_loaders(
            forget_split,
            qa_formatter,
            dataset_name,
            n_perturbed,
            tokenizer,
            batch_size,
            random_seed,
        )
        logger.info("Creating retain loaders...")
        self.retain_loaders = self.get_eval_data_loaders(
            retain_split,
            qa_formatter,
            dataset_name,
            n_perturbed,
            tokenizer,
            batch_size,
            random_seed,
        )
        self.model = model
        self.base_truth_ratios = base_truth_ratios
        self.accelerator = accelerator
        self.n_print = n_print
        self.generate_kwargs = generate_kwargs
        self.tokenizer = tokenizer

    def evaluate(self) -> "EvaluateOutputs":
        """
        Run a complete evaluation on the forget and retain data, computing forget
        quality, model utility, and other associated prerequisite metrics.
        """
        logger.info("Evaluating on forget data...")
        forget_metrics = self.compute_dataset_metrics(
            self.model,
            self.forget_loaders,
            self.n_print,
            self.accelerator,
            **self.generate_kwargs,
        )
        logger.info("Evaluating on retain data...")
        retain_metrics = self.compute_dataset_metrics(
            self.model,
            self.retain_loaders,
            self.n_print,
            self.accelerator,
            **self.generate_kwargs,
        )
        logger.info("Calculating forget quality and model utility...")
        return self.forget_quality_model_utility(
            self.base_truth_ratios, forget_metrics, retain_metrics
        )

    @staticmethod
    def compute_dataset_metrics(
        model: PreTrainedModel,
        data_loaders: dict[str, DataLoader],
        n_print: int,
        accelerator: Accelerator = Accelerator(),
        **generate_kwargs: dict,
    ) -> dict[str, torch.Tensor]:
        """Computes losses, truth ratios, and rouge scores for a model on a dataset
        (forget or retain split).

        Args:
            model : Transformers model used to perform evaluation on
            data_loaders : Data loaders to use to perform evaluation (see
                get_eval_data_loaders)
            n_print : number of samples to print the generated answers for
            accelerator : Accelerator object to handle moving data etc. to device
            generate_kwargs : keyword arguments to pass to model.generate

        Returns:
            output dictionary with truth_ratios, all_losses, rougeL_recall, and
            rouge1_recall for each sample in the dataset
        """
        data_loader = data_loaders["full"]
        qa_loader = data_loaders["qa"]
        tokenizer: PreTrainedTokenizer = data_loader.dataset.tokenizer
        n_perturbed = data_loader.dataset.n_perturbed
        dataset_len = len(data_loader.dataset)
        batch_size = data_loader.batch_size
        output_dict = {
            "all_losses": torch.zeros(
                (dataset_len, n_perturbed + 1), dtype=torch.float64
            ),
            "truth_ratios": torch.zeros(dataset_len),
            "rougeL_recall": torch.zeros(dataset_len),
            "rouge1_recall": torch.zeros(dataset_len),
        }

        model, data_loader, qa_loader = accelerator.prepare(
            model, data_loader, qa_loader
        )

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
            output_dict["all_losses"][
                batch_start_index:batch_end_index, 0
            ] = gt_loss.cpu()

            # Perturbed answers
            for perturbed_index in range(n_perturbed):
                pt_batch = pt_batches[perturbed_index]
                with torch.no_grad():
                    # DO pass position_ids into this method -> see collator for info
                    p_output = model(**pt_batch)
                output_dict["all_losses"][
                    batch_start_index:batch_end_index, perturbed_index + 1
                ] = get_loss(p_output.logits, pt_batch["labels"]).cpu()

        output_dict["mean_loss_gt"] = torch.mean(output_dict["all_losses"][:, 0]).item()
        output_dict["mean_loss_perturbed"] = torch.mean(
            output_dict["all_losses"][:, 1:]
        ).item()

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
                    questions["input_ids"][:n_print_batch], skip_special_tokens=True
                )
                for q_text, gen_text, target_text in zip(
                    question_text,
                    generated_answers[:n_print_batch],
                    target_answers[:n_print_batch],
                ):
                    msg = (
                        f"\nQuestion: {q_text}\n\nTarget: {target_text}\n\n"
                        f"Generated: {gen_text}\n{'-' * 15}"
                    )
                    logger.info(msg)

            for rouge_idx, (generated_text, target_text) in enumerate(
                zip(generated_answers, target_answers)
            ):
                rouge_result = eval_rouge_recall(
                    gen_output=generated_text, ground_truth=target_text
                )
                output_dict["rougeL_recall"][batch_start_index + rouge_idx] = (
                    rouge_result["rougeL_recall"]
                )
                output_dict["rouge1_recall"][batch_start_index + rouge_idx] = (
                    rouge_result["rouge1_recall"]
                )

        # calculate truth_ratio and return them along with losses
        output_dict["truth_ratios"] = truth_ratio(output_dict["all_losses"])

        return output_dict

    @staticmethod
    def forget_quality_model_utility(
        base_truth_ratios: torch.Tensor | None,
        forget_metrics: dict[str, torch.Tensor],
        retain_metrics: dict[str, torch.Tensor],
    ) -> "EvaluateOutputs":
        """
        Compute the forget quality and model utility metrics for a model.

        Args:
            base_truth_ratios: base model truth ratios to compare against in the ks_test
                If None, the forget_quality metrics will not be calculated.
            forget_metrics: metrics from compute_dataset_metrics for the forget data
            retain_metrics: metrics from compute_dataset_metrics for the retain data

        Returns:
            EvaluateOutputs instance contatining the input forget and retain metrics
            (with keys forget_* and retain_*) as well as the following metrics:
                forget_quality_1 : 1-tailed ks-test comparison between forget and base
                    model truth ratios
                forget_quality_2 : 2-tailed ks-test comparison between forget and base
                    model truth ratios
                retain_mean_tr : mean of the retain truth ratios
                retain_mean_rougeL_recall : mean of the rougeL recall on the retain data
                retain_model_utility : harmonic mean of the retain mean truth ratio and
                    mean rougeL recall
        """
        results_dict = {}

        # --------------
        # Metrics on forget dataset (i.e. quality of forgetting)
        # --------------
        if base_truth_ratios is not None:
            # one-tailed: truth ratio CDF for forget is greater than the base model
            results_dict["forget_quality_1"] = np.log(
                ks_test(
                    forget_metrics["truth_ratios"],
                    base_truth_ratios,
                    alternative="greater",
                )
            ).item()
            # two-tailed: truth ratio CDF for forget is close to base model
            results_dict["forget_quality_2"] = np.log(
                ks_test(forget_metrics["truth_ratios"], base_truth_ratios)
            ).item()
        else:
            results_dict["forget_quality_1"] = None
            results_dict["forget_quality_2"] = None

        # raw mean for forget truth ratios (not clamped like retain below)
        results_dict["forget_mean_tr"] = torch.mean(
            forget_metrics["truth_ratios"]
        ).item()

        # --------------
        # Metrics on retain dataset (i.e. performance of model after forgetting)
        # --------------
        # need to transform truth ratios according to table 1 in the TOFU paper
        transform_retain_tr = torch.clamp((1 - retain_metrics["truth_ratios"]), 0)
        results_dict["retain_mean_tr"] = torch.mean(transform_retain_tr).item()

        results_dict["retain_mean_rougeL_recall"] = torch.mean(
            retain_metrics["rougeL_recall"]
        ).item()

        results_dict["retain_model_utility"] = hmean(
            [results_dict["retain_mean_tr"], results_dict["retain_mean_rougeL_recall"]]
        )

        return EvaluateOutputs.combine_outputs(
            forget_metrics, retain_metrics, results_dict
        )

    @staticmethod
    def get_eval_data_loaders(
        data_split: Dataset,
        qa_formatter,
        dataset_name,
        n_perturbed,
        tokenizer,
        batch_size,
        random_seed,
    ) -> dict[str, DataLoader]:
        """
        Create two data loaders for evaluation on a given data split, one that returns
        inputs for combined question and answers (ground truth and perturbed answers),
        and one that returns inputs for questions and answers separately
        (only for ground truth answers).

        Args:
            data_split, qa_formatter, n_perturbed, tokenizer, random_seed :
                See EvalQADataset for details
            batch_size : Batch size for both data loaders
        Returns:
            dict with "full" data loader (combined q + a for ground truth and perturbed)
            and "qa" data loader (separate q and a for ground truth only)
        """
        dataset = EvalQADataset(
            data_split,
            tokenizer,
            qa_formatter,
            n_perturbed=n_perturbed,
            dataset_name=dataset_name,
            random_seed=random_seed,
        )

        tokenizer.padding_side = "left"
        eval_collate_fn = EvaluateDataCollator(tokenizer=tokenizer)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=eval_collate_fn,
        )

        # Make data loader that returns batches of separated questions and ground truth
        # answers (so this can then be passed to accelerate to handle moving data to
        # device). batch[0] below selects the ground truth q/a pair (rather than
        # perturbed answers). batch size 1 as batch already created in previous data
        # loader
        qa = [extract_qa_for_generate(batch[0], tokenizer) for batch in data_loader]
        qa_loader = DataLoader(
            qa, batch_size=1, collate_fn=lambda x: x[0], shuffle=False
        )

        return {"full": data_loader, "qa": qa_loader}

    def __len__(self) -> int:
        return len(self.forget_loaders["full"].dataset) + len(
            self.retain_loaders["full"].dataset
        )


@dataclass
class EvaluateOutputs:
    forget_quality_1: float | None  # None if evaluation run without base_truth_ratios
    forget_quality_2: float | None  # None if evaluation run without base_truth_ratios
    forget_all_losses: torch.Tensor
    forget_truth_ratios: torch.Tensor
    forget_rougeL_recall: torch.Tensor
    forget_rouge1_recall: torch.Tensor
    forget_mean_loss_gt: float
    forget_mean_loss_perturbed: float
    forget_mean_tr: float  # raw mean of forget truth ratios
    retain_mean_tr: float  # clamped mean of 1 - retain truth ratios
    retain_mean_rougeL_recall: float
    retain_model_utility: float
    retain_all_losses: torch.Tensor
    retain_truth_ratios: torch.Tensor
    retain_rougeL_recall: torch.Tensor
    retain_rouge1_recall: torch.Tensor
    retain_mean_loss_gt: float
    retain_mean_loss_perturbed: float

    def save(self, path: str) -> None:
        """
        Save an EvaluateOutputs instance to a json file.
        """
        with open(path, "w") as f:
            json.dump(self.to_raw_dict(), f)

    def to_raw_dict(self) -> dict[str, float | list | None]:
        """
        Converts the EvaluateOutputs instance to a dictionary, converting any torch
        tensors or numpy arrays to lists for compatibility with saving as JSON.
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
                data[key] = torch.tensor(data[key], dtype=torch.double)
        return cls(**data)

    @classmethod
    def load(cls, path: str) -> "EvaluateOutputs":
        """
        Load an EvaluateOutputs instance from a json file.
        """
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    @property
    def summary_metrics(self) -> dict[str, float]:
        """
        Returns all single-valued forget quality/model utility metrics
        """
        return {
            k: v
            for k, v in asdict(self).items()
            if isinstance(v, (float, int)) or v is None
        }

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

    def __getitem__(self, key: str) -> float | torch.Tensor | None:
        # Some Trainer methods add eval_ prefix to metrics, so remove it if present here
        return getattr(self, key.removeprefix("eval_"))
