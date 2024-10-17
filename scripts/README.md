# README for Selective Forgetting Scripts

This README describes the configs and scripts required to run fine-tuning, forgetting, and evaluation jobs for the ARC Selective Forgetting project.

## Configs

### Top-Level Experiment Specification

At the parent level of `configs/experiment` are "top-level" config files that specify a whole group of experiments to run together (a set of full models, retain models, and forget models). For every combination of dataset, model-hyperparameter config pair, and full model and forget technique + hyperparameter combination, they will generate one individual job config (which we also call an experiment, confusingly).

You can see an example of a top experiment config at <configs/experiment/tofu_test.yaml>.

A top experiment config is a yaml file containing a **combinations** argument. Nested under this should be:

- **data_config**: A list containing named data configs in `configs/data`
- **train_config**: A list containing named hyperparameter sets to use for `full` and `retain` jobs (names of files in `configs/model/<model_config>/hyperparameters`)
- **forget_config**: A list of forget method and forget hyperparameter pairs to generate `forget` models for, for every `retain` model.
- **seed:** A list of seeds to generate experiments over

It should additionally contain:

- **model_config**: The base model config to use for all jobs - top level configs must only specify one base model to use for all jobs.
- **full_data_config:** The name of the data config in configs/data that defines the full fine-tuning dataset
- **use_bask:** Whether to create a slurm array job for running on baskerville (you can treat this as whether to generate a slurm script - see our template in <src/arcsf/config/jobscript_template.sh>)
- **bask:** Arguments used in generating the baskerville array job script. Only used if **use_bask** is true.
- **wandb_kwargs:** Contains **use_wandb** and **wandb_config**, as defined above.
- **model_cache_dir:** HuggingFace models cache path (where to download/look for base models)
- **data_cache_dir:** HuggingFace datasets cache path (where to download/look for datasets)
- **wandb_cache_dir:** WandB cache path (where to log WandB run outputs)


### Individual Experiment Configs (Job Config)

An individual experiment config defines one model training job and all its associated data, model, and training hyperparameters. The type of experiment is determined by the `train_type` parameter, as follows:

- `full`: Fine-tune a base model on a whole dataset, which normally should use a data config specifying a forget size of zero (such as <configs/data/tofu_full.yaml>). These models are used as the starting model for `forget` training jobs.
- `retain`: Fine-tune a base model on a retain dataset, a subset of the whole dataset excluding the data to forget. This is used as the gold standard to compare forget models to, as retain models have never seen the forget data.
- Forget jobs aim to remove the knowledge of a forget data subset using one of the `full` models as the starting point. Forget experiments are specified by setting `train_type` to the name of one of the unlearning techniques:
  - `ascent`: gradient ascent
  - `difference`: gradient difference
  - `idk`: I don't know
  - `kl`: KL divergence

Example experiment config:

```yaml
data_config: tofu_20_20
experiment_name: tofu_test/experiment_0
full_model_name: tofu_test/full_0
hyperparameter_config: shorter_noeval
model_config: gpt2
seed: 42
train_type: ascent
use_wandb: true
wandb_config:
  entity: turing-arc
  group: tofu-test
  log_model: 'false'
  project: selective-forgetting
```

All experiment configs contain:

- The train type (see above)
- The name of a dataset config stored in configs/data, e.g. <configs/data/tofu_20_20.yaml>
  - Note that if a full job, the data config needs to be set up such that the full dataset is used, e.g. <configs/data/tofu_full.yaml>
- The path to the relevant full model (or the path where the full model will be saved if the experiment is a full training job) - `full_model_name`
- This name of base model config, which should be present in <configs/model>, e.g. `gpt2` for <configs/model/gpt2/gpt2.yaml>
- The name of a hyperpameters config stored within the model folder's hyperapameter folder, e.g. <configs/model/gpt2/hyperparameters/longer.yaml>
- The name of the experiment (models from `retain` and `forget` jobs using the same full model and retain/forget split will normally be grouped under the same experiment output directory)
- **seed:** A random seed used for replicability
- (Optionally) **use_wandb:** A boolean determining whether or not to use wandb
- (Optionally) **wandb_config:** Contains arguments for wandb


### Data Configs

Must contain two main arguments:

- **dataset_name:** Name of the base dataset to be used, e.g. `tofu`. Should correspond to `_dataset_dict` in <src/arcsf/data/data_module.py>
- **data_kwargs:** Kwargs passed to `get_data`, as defined in <src/arcsf/data/data_module.py>

### Model Configs

Must contain the following arguments:

- **model_id:** Model ID as on HuggingFace (or a path to a local base model)
- **model_kwargs:** Kwargs passed to `AutoModelForCausalLM.from_pretrained()`, see <https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForCausalLM.from_pretrained>
- **add_padding_token:** Boolean determining whether to add a padding token IF one does not already exist in the tokenizer.
- **qa_formatter_kwargs:** Templates to format question-answer pairs:
  - **question_template:** Must contain `{question}`, which will be replaced with the question string. Can also contain a system prompt before the question, for example.
  - **answer_template:** Must contain `{answer}`, which will be replaced with the answer string. Can also contain end of text tags or other special tokens required by the model.

### Hyperparameter Configs

Contain the following arguments:

- **trainer_kwargs:** Arguments passed to `TrainingArguments`. See <https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments>
- [Optional] **peft_kwargs:** Keyword arguments passed to `LoraConfig` if using parameter-efficient fine-tuning See <https://huggingface.co/docs/peft/en/package_reference/lora>
- [Optional] **early_stopping_kwargs:** Early stopping kwargs. Optional. If provided, early stopping will be used. Should contain kwargs for `EarlyStoppingCallback`. See <https://huggingface.co/docs/transformers/en/main_classes/callback#transformers.EarlyStoppingCallback>


## Scripts

### Generating Individual Experiment Configs (Job Configs)

To generate all the individual training job configs and submission scripts for an experiment (see below for more detail on their format), you can run the `gen_configs` script, e.g.:

```bash
python scripts/gen_configs.py tofu_test
```

where `tofu_test` is the name of a top-level experiment config file in `configs/experiment`.

This generates:
- A directory `configs/experiment/tofu_test/` that contains all the training configs for the experiment (full models, retain models, and forget models).
- A directory `train_scripts/tofu_test/` that contains slurm scripts for submitting the jobs on Baskerville. These are only generated if the top-level experiment config contains Baskerville arguments.

### Submission Scripts

 The `train-scripts/<experiment_name>/` directory contains slurm scripts to submit full, retain, and forget model training jobs, and evaluation jobs. Forget jobs and full model evaluation jobs depend on the full and retain models having been trained first.

The `train-scripts/queue-all.sh` script can be used to put all the jobs on the queue at once, with the appropriate dependencies to ensure jobs are run in the correct order. It should be run from the parent `arc-selective-forgetting` directory as follows:

```bash
bash train-scripts/queue_all.sh tofu_test`
```

Alternatively, you can submit them individually with `sbatch`, e.g.

```bash
sbatch train_scripts/tofu_test/retain.sh
```

or run the training script manually (see below).

### Training Script

Training jobs (full, retain, and forget) are all done via `scripts/train.py` - this is what all the submission scripts call. If you'd like to run it manually, the script takes the name of a single (non-top) experiment training config stored in `configs/experiment/` (see below on configs) as input, e.g.

```bash
python scripts/train.py --experiment_name tofu_test/retain_0
```

You can also create your own individual job config and submit it this way (see below for the individual job config structure).

Training jobs will log metrics to WandB, if specified in the config file.

### Evaluation

Evaluation metrics for retain and forget model jobs are also saved by the training script. Full model evaluations are done separately, as they must be done for each retain/forget split of interest (which is not known at the point of training the full model on the whole dataset). This is done with the `scripts/full_eval.py` script, e.g.:

```bash
python scripts/full_eval.py --experiment_name tofu_test/retain_0
```

where `--experiment_name` is the name of an individual retain model config.

There's also a script `script/all_eval.py` that can be used to re-compute or generate additional evaluation metrics for other models. In particular it takes these arguments (which are also arguments to `full_eval` above):

- `--experiment_2_eval`: Set to run evaluations on the subset of the retain set that contains references to the entities whose relationship is being forgotten (see the report for more details)
- `--train_set_eval`: Run evaluations on the original phrasings of the questions and answers used in training, rather than paraphrased versions.


## Output Files

Output files (saved models and evaluations) are saved to `outputs/<experiment_name>/experiment_<idx>`, e.g. `outputs/tofu_test/experiment_0`. Each `experiment_<idx>` directory contains the output for one retain model (specified by a data config, model config, hyperparameter config, and random seed) and all its associated forget models (generated with the same data config, model config, and seed, but maybe with multiple forget techniques and hyperparameters). Each full model's outputs are saved to an `outputs/<experiment_name>/full_<idx>` directory.

Evaluation metrics are saved (and can be loaded) in JSON format by the `EvaluateOutputs` class in `arcsf.eval.evaluate` - see its docstring and the documentation of the `Evaluator` class for detail on its contents, as well as the project report (e.g. for details on how our evaluations differ to the original TOFU paper).
