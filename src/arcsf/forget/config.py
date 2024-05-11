import os
import warnings
from copy import copy

import wandb
import yaml

from arcsf.config.config_class import Config
from arcsf.models.config_class import ModelConfig


class ForgetConfig(Config):

    def __init__(
        self,
        data_config: str,
        model_config: str,
        model_path: str,
        forget_class: str,
        use_wandb: bool,
        wandb_config: dict | None,
        seed: int | None,
    ) -> None:
        super().__init__()

        # Load in other configs
        # TODO: replace data config loading
        with open(f"configs/data/{data_config}.yaml", "r") as f:
            self.data_config = yaml.safe_load(f)
        # self.data_config = None
        self.model_config = ModelConfig.from_yaml(f"configs/model/{model_config}.yaml")
        if model_path:
            self.model_config.model_id = model_path
        # Either "all" (train on full dataset) or "retain" (train on retain split only)
        self.forget_class = forget_class

        # Wandb args
        self.use_wandb = use_wandb
        self.wandb_config = wandb_config

        # Setup run name
        self.data_name = data_config
        self.model_name = model_config
        self.experiment_name = f"{data_config}-{model_config}-{forget_class}"

        # seed
        self.seed = seed

    @classmethod
    def from_dict(cls, config_dict) -> "ForgetConfig":
        """Create a ForgetConfig from a config dict.

        Args:
            config_dict: Dict containing "data_config", "model_config", "model_path",
                "forget_class", "seed", "use_wandb", "wandb_args" keys.

        Returns:
            FineTuningConfig object.
        """
        return cls(**config_dict)

    # TODO: complete this method
    def to_dict(self) -> dict:
        return {
            "experiment_name": self.experiment_name,
            "data_name": self.data_name,
            "data_config": self.data_config,
            "forget_class": self.forget_class,
            "model_config": self.model_config.to_dict(),
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

        wandb.init(
            config={"selective-forgetting": self.to_dict()},
            job_type=job_type,
            **wandb_config,
        )
