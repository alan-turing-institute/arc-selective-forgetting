# README for Selective Forgetting Scripts

TODO: flesh this out

## Training

currently done via `scripts/train.py`. Takes the name of a single (non-top) experiment config stored in `configs/experiment/` as input, e.g.

```bash
python scripts/train.py --experiment_name example_experiment_config
```

Alternatively if using baskerville (or another HPC that uses slurm for job submission), you can use slurm scripts to submit. E.g.

Using slurm:

```bash
sbatch scripts/example_slurm_script.sh
```

## Experiment Configs

defines for a single experiment which data arguments and model to use. later on will define an experiment end to end, including which forgetting techniques to apply.


## Top Experiment Configs

To convert a top experiment config into a collection of configs collected under `configs/experiments/top_config_name/`

```python
from arcsf.config.experiment import generate_experiment_configs
generate_experiment_configs("example_top_experiment_config")
```

If your top experiment config includes baskerville arguments, this will also generate submission scripts under `bask/top_config_name/`
