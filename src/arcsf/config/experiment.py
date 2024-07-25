import os
import warnings
from copy import deepcopy
from itertools import product
from pathlib import Path

import wandb
import yaml
from jinja2 import Environment, FileSystemLoader, Template

from arcsf.config.config import Config
from arcsf.constants import (
    DATA_CONFIG_DIR,
    EXPERIMENT_CONFIG_DIR,
    MODEL_CONFIG_DIR,
    PROJECT_DIR,
)
from arcsf.data.config import DataConfig
from arcsf.models.config import ModelConfig
from arcsf.utils import get_model_path


def _make_config_name(top_config_name: str, experiment_type: str, n: int) -> str:
    return f"{top_config_name}/{experiment_type}_{n}"


def _generate_combos_from_dict(combo_dict: dict) -> dict:
    """
    Takes a dictionary containing lists, and produces all combinations of the elements
    of those lists. If any entry is not a list, it is treated as a list of length 1.
    Adds train_type to each output combination depending on the 'full' argument.

    Args:
        combo_dict: Dictionary of lists to create combinations from
        full: If True, train_type in output combinations is 'full', else is 'retain'

    Returns:
        A dictionary containing all possible combinations
    """
    # Convert combo dict into dict of combos
    sweep_dict_keys, sweep_dict_vals = zip(*combo_dict.items())
    combinations = [
        dict(zip(sweep_dict_keys, v)) for v in product(*list(sweep_dict_vals))
    ]

    # Return
    return combinations


def generate_combos_from_dict(combo_dict: dict, wandb_kwargs: dict) -> dict:
    """
    Takes a dictionary containing lists, and produces all combinations of the elements
    of those lists. If any entry is not a list, it is treated as a list of length 1.
    Adds train_type to each output combination depending on the 'full' argument.

    Args:
        combo_dict: Dictionary of lists to create combinations from
        full: If True, train_type in output combinations is 'full', else is 'retain'
        wandb_kwargs: A dictionary of wandb arguments added to each combination

    Returns:
        A dictionary containing all possible combinations
    """
    # Generate combinations
    combinations = _generate_combos_from_dict(combo_dict)

    # Process model kwargs
    for n, _ in enumerate(combinations):
        combinations[n] = {**combinations[n], **wandb_kwargs}

    # Return
    return combinations


def make_config(
    base_config: dict,
    data_config: str,
    model_config: str,
    train_type: str,
    full_model_name: str,
    hyperparameter_config: str,
    experiment_name: str,
) -> dict:
    config = deepcopy(base_config)
    config["data_config"] = data_config
    config["full_model_name"] = full_model_name
    config["model_config"] = model_config
    config["train_type"] = train_type
    config["hyperparameter_config"] = hyperparameter_config
    config["experiment_name"] = experiment_name
    return config


def write_yaml(data, path):
    with open(path, "w") as f:
        yaml.dump(data, f)


def write_train_script(
    template: Template,
    top_config_name: str,
    job_type: str,
    bask_config: dict,
    array_number: int,
    script_dir: Path,
):
    train_script = template.render(
        job_name=f"{top_config_name}_{job_type}",
        walltime=bask_config["walltime"],
        node_number=bask_config["node_number"],
        gpu_number=bask_config["gpu_number"],
        array_number=array_number,
        script_name="scripts/train.py",
        experiment_file=f"{top_config_name}/{job_type}",
    )
    # Create directory for train scripts if it doesn't exist
    save_dir = script_dir / top_config_name
    if not save_dir.is_dir():
        os.makedirs(save_dir)

    with open(save_dir / f"{job_type}.sh", "w") as f:
        f.write(train_script)


def generate_experiment_configs(top_config_name: str) -> None:
    """
    Takes a top config and generates all experiments from combinations of data configs,
    model/hyperparameter config pairs, and seeds.

    The top config structure is described in more detail in scripts/README, but broadly
    it is expected to contain a combinations argument with the following nested
    underneath:

    - data_config: a list of data configs
    - model_config: a list of [model_config, hyperparameter_config] pairs
    - seeds: a list of seeds

    It should also contain a full fine-tuning data config, wandb kwargs (if desired),
    and baskerville kwargs (if desired).

    Args:
        top_config_name: The name of a top config YAML file in configs/experiments
    """

    # Read in yaml file
    with open(EXPERIMENT_CONFIG_DIR / f"{top_config_name}.yaml") as f:
        top_config = yaml.safe_load(f)

    # Get wandb kwargs
    wandb_kwargs = {}
    if "wandb_kwargs" in top_config.keys():
        wandb_kwargs = {**top_config["wandb_kwargs"]}

    # Generate combinations
    data_configs = top_config["combinations"].pop("data_config")
    forget_combos = top_config["combinations"].pop("forget_config")
    combinations = generate_combos_from_dict(top_config["combinations"], wandb_kwargs)
    n_full = 0
    n_retain = 0
    n_forget = 0

    outdir = EXPERIMENT_CONFIG_DIR / top_config_name
    if not outdir.is_dir():
        os.makedirs(outdir)

    for full_idx, c in enumerate(combinations):  # Loop over seeds, full/retain hparams
        # Full model config
        full_model_name = _make_config_name(top_config_name, "full", full_idx)
        train_hparams = c.pop("train_config")
        flc = make_config(
            base_config=c,
            data_config=top_config["full_data_config"],
            model_config=top_config["model_config"],
            train_type="full",
            full_model_name=full_model_name,
            hyperparameter_config=train_hparams,
            experiment_name=full_model_name,
        )
        write_yaml(flc, EXPERIMENT_CONFIG_DIR / f"{full_model_name}.yaml")
        n_full += 1

        for dc in data_configs:  # Loop over forget/retain splits
            # Retain model config
            experiment_name = _make_config_name(top_config_name, "experiment", n_retain)
            retain_name = _make_config_name(top_config_name, "retain", n_retain)
            rtc = make_config(
                base_config=c,
                data_config=dc,
                model_config=top_config["model_config"],
                train_type="retain",
                full_model_name=full_model_name,
                hyperparameter_config=train_hparams,
                experiment_name=experiment_name,
            )
            write_yaml(rtc, EXPERIMENT_CONFIG_DIR / f"{retain_name}.yaml")
            n_retain += 1

            for fgc in forget_combos:  # Loop over forget methods & hyperparameters
                # Forget model config
                forget_name = _make_config_name(top_config_name, "forget", n_forget)
                fgc = make_config(
                    base_config=c,
                    data_config=dc,
                    model_config=top_config["model_config"],
                    train_type=fgc[0],
                    full_model_name=full_model_name,
                    hyperparameter_config=fgc[1],
                    experiment_name=experiment_name,
                )
                write_yaml(fgc, EXPERIMENT_CONFIG_DIR / f"{forget_name}.yaml")
                n_forget += 1

    # Check this is a reasonable number of jobs for an array of training jobs
    if n_forget > 1001:
        warnings.warn("Slurm array jobs cannot exceed more than 1001!")

    # Check whether to generate baskerville scripts:
    if top_config["use_bask"]:
        # Get jinja template
        environment = Environment(
            loader=FileSystemLoader(PROJECT_DIR / "src" / "arcsf" / "config")
        )
        template = environment.get_template("jobscript_template.sh")
        script_dir = PROJECT_DIR / "train_scripts"

        # Generate and save train scripts
        for job_type, n_jobs in zip(
            ["full", "retain", "forget"], [n_full, n_retain, n_forget]
        ):
            write_train_script(
                template=template,
                top_config_name=top_config_name,
                job_type=job_type,
                bask_config=top_config["bask"],
                array_number=n_jobs - 1,
                script_dir=script_dir,
            )


class ExperimentConfig(Config):
    """Experiment config class.

    Attributes:
        data_config: Dataset config class.
        model_config: Model config class, covering both model and hyperparameters.
        train_type: Either "retain" or "full": whether fine-tuning on a full dataset
                    or a retain set.
        full_model_name: If train_type is "retain", the name of the experiment yaml
                         file used for fine-tuning the full model to be compared against
        use_wandb: Whether or not to use wandb for logging
        wandb_config: Dict of wandb arguments
        run_name: Name of the run logged to wandb
        seed: Seed for random generation
    """

    def __init__(
        self,
        experiment_name: str,
        data_config: str,
        model_config: str,
        hyperparameter_config: str,
        train_type: str,
        use_wandb: bool,
        wandb_config: dict | None,
        seed: int | None,
        full_model_name: str | None = None,
    ) -> None:
        super().__init__()
        self.experiment_name = experiment_name
        self.config_names = {
            "data_config": data_config,
            "model_config": model_config,
            "hyperparameter_config": hyperparameter_config,
        }
        # Load in other configs
        self.data_config = DataConfig.from_yaml(DATA_CONFIG_DIR / f"{data_config}.yaml")
        model_dir = MODEL_CONFIG_DIR / model_config
        model_path = (
            str(get_model_path(full_model_name, "full"))
            if train_type not in ["full", "retain"]
            else None
        )

        self.model_config = ModelConfig.from_yaml(
            model_dir / f"{model_config}.yaml",
            model_path,
            model_dir / "hyperparameters" / f"{hyperparameter_config}.yaml",
        )
        # Check kwargs optional for full are present if doing retain/forget tuning
        if train_type != "full" and full_model_name is None:
            raise ValueError(
                "If train_type is retain or a forget method, full_model must be type "
                "str and is currently None"
            )

        # Either "full" (train on full dataset), "retain" (train on retain split only)
        # or the name of a forget method
        self.train_type = train_type

        # If this is a retain or forget model, this points to the full model against
        # which the retain model should be compared
        self.full_model_name = full_model_name

        # Wandb args
        self.use_wandb = use_wandb
        self.wandb_config = wandb_config

        # Setup run name
        self.run_name = f"{model_config}-{data_config}-{train_type}-{seed}"

        # seed
        self.seed = seed
        self.model_config.trainer_kwargs["seed"] = seed

    @classmethod
    def from_dict(cls, dict) -> "ExperimentConfig":
        """Create an ExperimentConfig from a config dict.

        Args:
            config: Dict that must contain all kwargs required to initialise an
                    ExperimentConfig.

        Returns:
            ExperimentConfig object.
        """
        return cls(**dict)

    def to_dict(self) -> dict:
        """Convert the config class into a dictionary format. Useful for saving.

        Returns:
            Dictionary containing the the attributes of the ExperimentConfig,
            expect those relating to wandb.
        """
        return {
            "experiment_name": self.experiment_name,
            "config_names": self.config_names,
            "train_type": self.train_type,
            "data_config": self.data_config.to_dict(),
            "model_config": self.model_config.to_dict(),
            "full_model_name": self.full_model_name,
            "seed": self.seed,
        }

    def save(self, path: str) -> None:
        """Save the config class into a yaml file.

        Args:
            path: Path and file name under which to save the config class.
        """
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f)

    def init_wandb(self, job_type: str) -> None:
        """Initialise a wandb run if the config specifies to use wandb and a run has not
        already been initialised.

        If name, group and job_type and not specificied in the input config then they
        are set as:
                name: run_name
                group: data_set_name_config_gen_dtime OR data_set_name
                job_type: misc
        """
        if not self.use_wandb:
            warnings.warn("Ignored wandb initialisation as use_wandb=False")
            return
        if wandb.run is not None:
            raise ValueError("A wandb run has already been initialised")

        wandb.login()
        wandb_config = deepcopy(self.wandb_config)

        if "log_model" in wandb_config:
            # log_model can only be specified as an env variable, so we set the env
            # variable then remove it from the init args.
            os.environ["WANDB_LOG_MODEL"] = wandb_config["log_model"]
            wandb_config.pop("log_model")

        # set default names for any that haven't been specified
        wandb_config["name"] = self.run_name
        wandb_config["job_type"] = job_type

        wandb.init(config={"selective-forgetting": self.to_dict()}, **wandb_config)
