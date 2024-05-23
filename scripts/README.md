# README for Selective Forgetting Scripts

This README describes the configs required to run fine-tuning, forgetting, and evaluation jobs for the ARC Selective Forgetting project.

## Training

Training jobs are done via `scripts/train.py`. This script takes the name of a single (non-top) experiment config stored in `configs/experiment/` (see below on configs) as input, e.g.

```bash
python scripts/train.py --experiment_name gpt2_longer_experiment_retain
```

Alternatively if using baskerville (or another HPC that uses slurm for job submission), you can use slurm scripts to submit. E.g.

Using slurm:

```bash
sbatch scripts/gpt2_longer_retain.sh
```

To convert a top experiment config into a collection of configs collected under `configs/experiments/top_config_name/`

```python
from arcsf.config.experiment import generate_experiment_configs
generate_experiment_configs("example_top_experiment_config")
```

If your top experiment config includes baskerville arguments, this will also a baskerville array job submission script under the name of `train_scripts/<top_config_name>.sh`. This can then be subitted to submit all generated fine-tuning jobs, e.g.:

```bash
sbatch train_scripts/example_top_experiment_config.sh
```

## Configs

### Experiment Configs

An experiment config defines for a single experiment what dataset, model, hyperamaters, and any other relevant arguments to use. A single 'experiment' here refers to one of two things:

- Fine-tuning a model on a retain dataset, performing forgetting on (another) model fine-tuned on the full dataset, evaluating the forget model(s) against the retain model as a gold standard. You can see an example at <configs/experiment/gpt2_longer_experiment_retain.yaml>
- Fine-tuning a model on the full dataset. One full model will typically be used for comparison for many retain models <configs/experiment/gpt2_longer_experiment_full.yaml>

All experiment config should contain the following:

- The name of a dataset config stored in configs/data, e.g. <configs/data/example_tofu_1.yaml>
  - Note that if a full job, the data config needs to be set up such that the full dataset is used, e.g. <configs/data/example_tofu_full.yaml>
- The name of a model folder stored in configs/model/
  - This folder should contain a model config of the same name, e.g. <configs/experiment/gpt2/gpt2.yaml>
- The name of a hyperpameters config stored within the model folder's hyperapameter folder, e.g. <configs/experiment/gpt2/hyperparameters/longer.yaml>

The following additional arguments should be contained in the config:

- **seed:** A random seed used for replicability
- **use_wandb:** A boolean determining whether or not to use wandb
- **wandb_config:** Contains arguments for wandb

If the experiment config is for a retain job, it should additionally contain:

- **full_model_name:** The name of another experiment config which is a full model fine-tuning config

### Top Experiment Configs

Top experiment configs are used to generate a large number of jobs together. For every combination of dataset, model-hyperparameter config pair, and seed, they will generate one job.

**Note:** Since the top experiment config will generate both retain and full fine-tuning jobs, you should ensure that:

1. The top config covers all combinations for a given base dataset (e.g. tofu), model-hyperameter, and seed combination to avoid duplication of full fine-tuning jobs
2. The top config should contain only one full fine-tuning job, so you should not mix base datasets (e.g. tofu and another dataset) in the same top config

You can see an example of a top experiment config at <configs/experiment/example_top_experiment_config.yaml>.

A top experiment config be a yaml file containing a **combinations** argument. Nested under this should be:

- **data_config**: A list containing of named data configs in configs/data
- **model_config**: A list containing lists of paired model configs and an associated hyperparameter config (see experiment config above for file structure)
- **seed:** A list of seeds to generate experiments over

It should additionally contain:

- **full_data_config:** The name of the data config in configs/data that defines the full fine-tuning dataset
- **use_bask:** Whether to create a slurm array job for running on baskerville (you can treat this as whether to generate a slurm script - see our template in <src/arcsf/config/jobscript_template.sh>)
- **bask:** Arguments used in generating the baskerville array job script. Only used if **use_bask** is true.
- **wandb_kwargs:** Contains **use_wandb** and **wandb_config**, as defined above.

### Data Configs

Must contain two main arguments:

- **dataset_name:** Name of the base dataset to be used, e.g. `tofu`. Should correspond to `_dataset_dict` in <src/arcsf/data/data_module.py>
- **data_kwargs:** Kwargs passed to `get_data`, as defined in <src/arcsf/data/data_module.py>

### Model Configs

Must contain the following arguments:

- **model_id:** Model ID as on HuggingFace
- **model_kwargs:** Kwargs passed to `AutoModelForCausalLM.from_pretrained()`, see <https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForCausalLM.from_pretrained>
- **add_padding_token:** Boolean determining whether to add a padding token IF one does not already exist in the tokenizer.

### Hyperparameter Configs

Must contain the following arguments:

- **trainer_kwargs:** Arguments passed to `TrainingArguments`. See <https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments>
- **early_stopping_kwargs:** Early stopping kwargs. Optional. If provided, early stopping will be used. Should contain kwags for `EarlyStoppingCallback`. See <https://huggingface.co/docs/transformers/en/main_classes/callback#transformers.EarlyStoppingCallback>
