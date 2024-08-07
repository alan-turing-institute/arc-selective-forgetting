import argparse

import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from arcsf.data.data_module import BlankQAFormatter, get_data
from arcsf.eval.evaluate import EvaluateOutputs, Evaluator
from arcsf.eval.plot import plot_cdf

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
    qa_formatter = BlankQAFormatter()
    loss_type = "standard"
    n_perturbed = 2
    random_seed = exp_config["seed"]

    # get splits
    forget_split, retain_split = get_data(
        exp_config["data_config"]["dataset_name"],
        **exp_config["data_config"]["data_kwargs"],
        random_seed=random_seed,
    )
    compare_eval = EvaluateOutputs.load(args.base_vals_path)

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
        args.eval_batch_size,
        max_new_tokens=50,
    )

    eval_results = evaluator.evaluate()
    eval_results.save(f"{model_dir}/eval_outputs.json")

    exp_name = exp_config["experiment_name"]
    print(f"\nBase Model path: {args.base_vals_path}")
    print(f"Test Model path: {model_dir}")
    print(f"Experiment Name: {exp_name}")
    print(eval_results)

    if args.plot:
        plot_cdf(
            compare_eval.forget_truth_ratio,
            eval_results.forget_truth_ratios,
            model_dir,
            exp_name,
            "forget",
        )
        plot_cdf(
            compare_eval.retain_truth_ratios,
            eval_results.retain_truth_ratios,
            model_dir,
            exp_name,
            "retain",
        )
