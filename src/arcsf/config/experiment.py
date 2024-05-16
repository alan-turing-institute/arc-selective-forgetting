import os
import warnings
from copy import copy
from itertools import product

import wandb
import yaml
from jinja2 import Environment, FileSystemLoader

from arcsf.config.config import Config
from arcsf.constants import (
    DATA_CONFIG_DIR,
    EXPERIMENT_CONFIG_DIR,
    MODEL_CONFIG_DIR,
    PROJECT_DIR,
)
from arcsf.data.config import DataConfig
from arcsf.models.config import ModelConfig


def _listify(obj: object):
    """Takes an object. If a list, returns it. If not, returns a list containing
    the object.

    Args:
        obj: Object to ensure is a list.

    Returns:
        Either the original object if a list or a list containing the object.
    """
    if isinstance(obj, list):
        return obj
    return [obj]


def generate_experiment_configs(
    top_config_name: str,
) -> None:

    # Read in yaml file
    with open(os.path.join(EXPERIMENT_CONFIG_DIR, f"{top_config_name}.yaml")) as f:
        top_config = yaml.safe_load(f)

    # Loop over, construct combo dict
    combo_dict = {}
    for key, value in top_config["configs"].items():
        combo_dict[key] = _listify(value)

    # Construct full set of combinations
    sweep_dict_keys, sweep_dict_vals = zip(*combo_dict.items())
    combo_dicts = [
        dict(zip(sweep_dict_keys, v)) for v in product(*list(sweep_dict_vals))
    ]

    # Check this is a reasonably number of jobs for an array of training jobs
    if len(combo_dicts) > 1001:
        warnings.warn("Slurm array jobs cannot exceed more than 1001!")

    # Check whether to generate baskerville scripts:
    use_bask = False
    if top_config["use_bask"]:
        use_bask = True
        if not os.path.exists("bask"):
            os.mkdir("bask")
        baskdir = f"bask/{top_config_name}"
        if not os.path.exists(baskdir):
            os.mkdir(baskdir)

    # Write out dicts and optionally bask scripts
    outdir = os.path.join(EXPERIMENT_CONFIG_DIR, f"{top_config_name}")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    for n, combo in enumerate(combo_dicts):
        file_name = f"{outdir}/run_{n}.yaml"
        with open(file_name, "w") as f:
            yaml.dump(combo, f)
        if use_bask:
            environment = Environment(
                loader=FileSystemLoader(
                    os.path.join(PROJECT_DIR, "src", "arcsf", "config")
                )
            )
            template = environment.get_template("jobscript_template.sh")

            script_content = template.render(
                job_name=f"submit_{n}",
                walltime=top_config["bask"]["walltime"],
                node_number=top_config["bask"]["node_number"],
                gpu_number=top_config["bask"]["gpu_number"],
                cpu_per_gpu=top_config["bask"]["cpu_per_gpu"],
                # array_number=, #TODO
                script_name="scripts/train.py",
                experiment_file=file_name,
            )

            with open(os.path.join(PROJECT_DIR, baskdir, f"submit_{n}.sh"), "w") as f:
                f.write(script_content)


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
        experiment_name: Name of the yaml file used to generate this config
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

        # Load in other configsg
        self.data_config = DataConfig.from_yaml(
            os.path.join(DATA_CONFIG_DIR, f"{data_config}.yaml")
        )
        model_dir = os.path.join(MODEL_CONFIG_DIR, model_config)
        self.model_config = ModelConfig.from_yaml(
            os.path.join(model_dir, f"{model_config}.yaml"),
            os.path.join(model_dir, "hyperparameters", f"{hyperparameter_config}.yaml"),
        )

        # TODO on another PR: if train_type == "retain", require forget + eval configs
        # otherwise if train_type == "full", these can be optional
        # Check kwargs optional for full are present if doing retain tuning
        if train_type == "retain":
            if full_model_name is None:
                raise ValueError(
                    "If train_type is retain, full_model must be type str and is "
                    "currently None"
                )

        # Either "all" (train on full dataset) or "retain" (train on retain split only)
        self.train_type = train_type

        # If this is a retain model, this points to the full model against which
        # the retain model should be compared
        self.full_model_name = full_model_name

        # Wandb args
        self.use_wandb = use_wandb
        self.wandb_config = wandb_config

        # Setup run name
        self.experiment_name = experiment_name

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
        wandb_config = copy(self.wandb_config)

        if "log_model" in wandb_config:
            # log_model can only be specified as an env variable, so we set the env
            # variable then remove it from the init args.
            os.environ["WANDB_LOG_MODEL"] = wandb_config["log_model"]
            wandb_config.pop("log_model")

        # set default names for any that haven't been specified
        wandb_config["name"] = f"{self.experiment_name}-{job_type}"

        wandb.init(config={"selective-forgetting": self.to_dict()}, **wandb_config)
