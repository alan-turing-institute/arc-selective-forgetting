import argparse
import logging
import os
import yaml

from arcsf.config.experiment import EXPERIMENT_CONFIG_DIR, ExperimentConfig
from arcsf.data.data_module import QAFormatter, get_data
from arcsf.eval.evaluate import EvaluateOutputs, Evaluator
from arcsf.models.model import load_model_and_tokenizer
from arcsf.utils import get_model_path

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=("Runs evaluation, for full models."))
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=False,
        help="Path to an experiment config file (specify this or model_dir)"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=False,
        help="Path to a model output directory (specify this or experiment_name)"
    )
    parser.add_argument(
        "--experiment_2_eval", action="store_true", help="Running experiment 2 eval."
    )
    parser.add_argument(
        "--train_set_eval", action="store_true", help="Eval on train set."
    )
    args = parser.parse_args()

    if (
        (args.model_dir and args.experiment_name) or
        (not args.model_dir and not args.experiment_name)
    ):
        raise RuntimeError("Specify one (only) of model_dir and experiment_name")

    if args.experiment_name:
        experiment_name = args.experiment_name
        experiment_config = ExperimentConfig.from_yaml(
            EXPERIMENT_CONFIG_DIR / f"{experiment_name}.yaml"
        )
        train_type = experiment_config.train_type
        seed = experiment_config.seed
        target_model_dir = get_model_path(experiment_name, train_type)

        data_config = experiment_config.data_config
        dataset_name = data_config.dataset_name
        dataset_kwargs = data_config.data_kwargs
        data_config_name = experiment_config.config_names["data_config"]

        model_config = experiment_config.model_config
        model_kwargs = model_config.model_kwargs
        add_padding_token = model_config.add_padding_token
        qa_formatter_kwargs = model_config.qa_formatter_kwargs
        trainer_kwargs = model_config.trainer_kwargs

    else:  # model_dir specified
        target_model_dir = args.model_dir
        experiment_config = yaml.safe_load(
            open(f"{target_model_dir}/experiment_config.yaml")
        )
        experiment_name = experiment_config["experiment_name"]
        train_type = experiment_config["train_type"]
        seed = experiment_config["seed"]
        
        data_config = experiment_config["data_config"]
        dataset_name = data_config["dataset_name"]
        dataset_kwargs = data_config["data_kwargs"]
        if "type" not in dataset_kwargs:
            # experiment 1 jobs (prior to experiment 2 loading implementation) did not
            # specify a split type in their saved data config
            logger.warning(
                "Defaulting split type to granularity due to missing value in saved "
                "data config"
            )
            dataset_kwargs["type"] =  "granularity"
        data_config_name = experiment_config["config_names"]["data_config"]

        model_config = experiment_config["model_config"]
        model_kwargs = model_config["model_kwargs"]
        add_padding_token = model_config["add_padding_token"]
        qa_formatter_kwargs = model_config["qa_formatter_kwargs"]
        trainer_kwargs = model_config["trainer_kwargs"]
    
    if train_type == "full":
        raise ValueError("Use full_eval.py for evaluating full models")

    if args.experiment_2_eval:
        dataset_kwargs["retain_subset"] = True

    print(f"Target model path: {target_model_dir}")

    # load model
    model, tokenizer = load_model_and_tokenizer(
        model_id=target_model_dir,
        peft_kwargs=None,  # don't need to add a new peft adapter for evals
        **model_kwargs,
        add_padding_token=add_padding_token,
    )
    batch_size = trainer_kwargs["per_device_eval_batch_size"]
    qa_formatter = QAFormatter(**qa_formatter_kwargs)
    n_perturbed = 3

    # get data splits
    forget_split, retain_split = get_data(
        dataset_name, **dataset_kwargs, random_seed=seed
    )

    # load base truth ratios for forget quality test from corresponding retain model
    if train_type != "retain":
        retain_model_dir = get_model_path(experiment_name, "retain")
        if args.experiment_2_eval:
            compare_path = (
                f"{retain_model_dir}/eval_outputs/{data_config_name}/entity_subset_eval"
                "/eval_outputs.json"
            )
        elif args.train_set_eval:
            compare_path = (
                f"{retain_model_dir}/eval_outputs/{data_config_name}/"
                "train_set_eval_outputs.json"
            )
        elif os.path.exists(
            f"{retain_model_dir}/eval_outputs/{data_config_name}/eval_outputs.json"
        ):
            compare_path = (
                f"{retain_model_dir}/eval_outputs/{data_config_name}/eval_outputs.json"
            )
        else:
            compare_path = f"{retain_model_dir}/eval_outputs.json"
        compare_eval = EvaluateOutputs.load(compare_path)
        compare_truth_ratios = compare_eval.forget_truth_ratios
    else:
        retain_model_dir = None
        compare_truth_ratios = None

    evaluator = Evaluator(
        model,
        forget_split,
        retain_split,
        qa_formatter,
        dataset_name,
        tokenizer,
        n_perturbed,
        seed,
        compare_truth_ratios,
        batch_size,
        train_set_eval=args.train_set_eval,
        max_new_tokens="adaptive",
    )

    eval_results = evaluator.evaluate()

    save_dir = f"{target_model_dir}/eval_outputs/{dataset_name}/"
    if args.experiment_2_eval:
        save_dir = f"{save_dir}/entity_subset_eval/"
    os.makedirs(save_dir, exist_ok=True)

    if args.train_set_eval:
        eval_results.save(f"{save_dir}/train_set_eval_outputs.json")
    else:
        eval_results.save(f"{save_dir}/eval_outputs.json")

    print(f"\nBase Model path: {retain_model_dir}")
    print(f"Test Model path: {target_model_dir}")
    print(f"Experiment Name: {experiment_name}")
    print(eval_results)
