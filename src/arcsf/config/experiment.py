import os
import warnings
from copy import copy
from itertools import product

import wandb
import yaml
from jinja2 import Environment, FileSystemLoader

from arcsf.config.config import Config
from arcsf.models.config import ModelConfig
from arcsf.utils import PROJECT_DIR


# Constants
def _get_project_dir(location):
    return os.path.join(PROJECT_DIR, location)


CONFIG_DIR = _get_project_dir("configs")
EXPERIMENT_CONFIG_DIR = _get_project_dir("experiment")
MODEL_CONFIG_DIR = _get_project_dir("model")
DATA_CONFIG_DIR = _get_project_dir("data")


def _listify(obj: object):
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

    def __init__(
        self,
        data_config: str,
        model_config: str,
        train_type: str,
        use_wandb: bool,
        wandb_config: dict | None,
        seed: int | None,
    ) -> None:
        super().__init__()

        # Load in other configs
        # TODO: replace data config loading
        with open(os.path.join(DATA_CONFIG_DIR, f"{model_config}.yaml"), "r") as f:
            self.data_config = yaml.safe_load(f)
        self.model_config = ModelConfig.from_yaml(
            os.path.join(MODEL_CONFIG_DIR, f"{model_config}.yaml")
        )

        # Either "all" (train on full dataset) or "retain" (train on retain split only)
        self.train_type = train_type

        # Wandb args
        self.use_wandb = use_wandb
        self.wandb_config = wandb_config

        # Setup run name
        self.data_name = data_config
        self.model_name = model_config
        self.experiment_name = f"{model_config}-{data_config}-{seed}"

        # seed
        self.seed = seed
        self.model_config.trainer_kwargs["seed"] = seed

    @classmethod
    def from_dict(cls, dict) -> "ModelConfig":
        """Create a FineTuningConfig from a config dict.

        Args:
            config: Dict that must contain "model_id", "model_kwargs", and
                "trainer_kwargs" keys.

        Returns:
            FineTuningConfig object.
        """
        return cls(**dict)

    # TODO: complete this method
    def to_dict(self) -> dict:
        return {
            "experiment_name": self.experiment_name,
            "data_name": self.data_name,
            "data_config": self.data_config,
            "train_type": self.train_type,
        }

    def save(self, path: str) -> None:
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
