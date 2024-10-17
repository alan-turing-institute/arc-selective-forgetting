# README for Selective Forgetting Scripts

This README describes the scripts required to run fine-tuning, forgetting, and evaluation jobs for the ARC Selective Forgetting project. The format of the config files is described in the [configs readme](../configs/README.md).

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
