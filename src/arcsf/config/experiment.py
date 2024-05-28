import os
import warnings
from copy import deepcopy
from itertools import product

import wandb
import yaml
from jinja2 import Environment, FileSystemLoader

import arcsf.constants
from arcsf.config.config import Config
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


def _make_config_name(top_config_name: str, n: int) -> str:
    return f"{top_config_name}/experiment_{n}"


def _generate_combos_from_dict(combo_dict: dict, full: bool) -> dict:
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

    # Initialise train_type
    train_type = "retain"
    if full:
        train_type = "full"

    # Add train_type to each combination
    for n, _ in enumerate(combinations):
        combinations[n]["train_type"] = train_type

    # Return
    return combinations


def generate_combos_from_dict(
    combo_dict: dict,
    full: bool,
    wandb_kwargs: dict,
) -> dict:
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
    combinations = _generate_combos_from_dict(combo_dict, full)

    # Process model kwargs
    for n, _ in enumerate(combinations):
        combinations[n]["hyperparameter_config"] = combinations[n]["model_config"][1]
        combinations[n]["model_config"] = combinations[n]["model_config"][0]
        combinations[n] = {**combinations[n], **wandb_kwargs}

    # Return
    return combinations


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
    with open(arcsf.constants.EXPERIMENT_CONFIG_DIR / f"{top_config_name}.yaml") as f:
        top_config = yaml.safe_load(f)

    # Loop over, construct combo dict
    combo_dict = {}
    for key, value in top_config["combinations"].items():
        combo_dict[key] = _listify(value)

    # Construct same dict with full dataset
    combo_dict_full = deepcopy(combo_dict)
    combo_dict_full["data_config"] = [top_config["full_data_config"]]

    # Get wandb kwargs
    wandb_kwargs = {}
    if "wandb_kwargs" in top_config.keys():
        wandb_kwargs = {**top_config["wandb_kwargs"]}

    # Get combinations
    full_combinations = generate_combos_from_dict(combo_dict_full, True, wandb_kwargs)
    retain_combinations = generate_combos_from_dict(combo_dict, False, wandb_kwargs)

    # Add full config path to
    for n, retain_combo in enumerate(retain_combinations):
        for m, full_combo in enumerate(full_combinations):
            if (
                full_combo["model_config"] == retain_combo["model_config"]
                and full_combo["seed"] == retain_combo["seed"]
                and full_combo["hyperparameter_config"]
                == retain_combo["hyperparameter_config"]
            ):
                retain_combinations[n]["full_model_name"] = _make_config_name(
                    top_config_name, m
                )

    # All combinations to generate yaml files for
    all_combinations = full_combinations + retain_combinations

    # Check this is a reasonable number of jobs for an array of training jobs
    if len(all_combinations) > 1001:
        warnings.warn("Slurm array jobs cannot exceed more than 1001!")

    # Write out dicts and optionally bask scripts
    outdir = os.path.join(arcsf.constants.EXPERIMENT_CONFIG_DIR, top_config_name)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    for n, combo in enumerate(all_combinations):
        file_name = (
            f"{arcsf.constants.EXPERIMENT_CONFIG_DIR}/"
            f"{_make_config_name(top_config_name, n)}.yaml"
        )
        with open(file_name, "w") as f:
            yaml.dump(combo, f)

    # Check whether to generate baskerville scripts:
    if top_config["use_bask"]:

        # Create directory for train script for experiment if it doesn't exist
        train_dir = arcsf.constants.PROJECT_DIR / "train_scripts"
        if not train_dir.is_dir():
            os.mkdir(train_dir)

        # Get jinja template
        environment = Environment(
            loader=FileSystemLoader(
                os.path.join(arcsf.constants.PROJECT_DIR, "src", "arcsf", "config")
            )
        )
        template = environment.get_template("jobscript_template.sh")

        # Generate and save train script
        job_name = f"{top_config_name}_train"
        train_script = template.render(
            job_name=job_name,
            walltime=top_config["bask"]["walltime"],
            node_number=top_config["bask"]["node_number"],
            gpu_number=top_config["bask"]["gpu_number"],
            array_number=len(all_combinations) - 1,
            script_name="scripts/train.py",
            experiment_file=f"{top_config_name}/experiment",
        )
        with open(os.path.join(train_dir, f"{job_name}.sh"), "w") as f:
            f.write(train_script)


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

        # Load in other configs
        self.data_config = DataConfig.from_yaml(
            arcsf.constants.DATA_CONFIG_DIR / f"{data_config}.yaml"
        )
        model_dir = arcsf.constants.MODEL_CONFIG_DIR / model_config
        self.model_config = ModelConfig.from_yaml(
            model_dir / f"{model_config}.yaml",
            model_dir / "hyperparameters" / f"{hyperparameter_config}.yaml",
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
        self.run_name = f"{model_config}-{data_config}-{seed}"

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
