import yaml

from arcsf.config.config import Config


class ModelConfig(Config):

    def __init__(
        self,
        model_id: str,
        model_kwargs: dict,
        trainer_kwargs: dict,
        early_stopping_kwargs: dict | None,
        peft_kwargs: dict | None,
        add_padding_token: bool = False,
    ) -> None:
        super().__init__()

        # Process main inputs
        self.model_id = model_id
        self.model_kwargs = model_kwargs
        self.peft_kwargs = peft_kwargs
        self.early_stopping_kwargs = early_stopping_kwargs
        self.add_padding_token = add_padding_token

        # Process trainer kwargs
        if isinstance(trainer_kwargs["learning_rate"], str):
            trainer_kwargs["learning_rate"] = float(trainer_kwargs["learning_rate"])
        trainer_kwargs["logging_dir"] = f'{trainer_kwargs["output_dir"]}/logs'
        if "logging_steps" in trainer_kwargs.keys():
            # Already the same by default, but this is to coerce it
            trainer_kwargs["save_steps"] = trainer_kwargs["logging_steps"]
            trainer_kwargs["eval_steps"] = trainer_kwargs["eval_steps"]

        # Add trainer kwargs to self
        self.trainer_kwargs = trainer_kwargs

        # Output dir
        self.output_dir = trainer_kwargs["output_dir"]

    @classmethod
    def from_yaml(cls, model_path: str, hyperparameter_path: str) -> "Config":
        with open(model_path, "r") as f:
            model_config = yaml.safe_load(f)
        with open(hyperparameter_path, "r") as f:
            hyperparameter_config = yaml.safe_load(f)
        return cls.from_dict(model_config, hyperparameter_config)

    @classmethod
    def from_dict(cls, model_dict, hyperparameter_dict) -> "ModelConfig":
        """Create a FineTuningConfig from a config dict.

        Args:
            config: Dict that must contain "model_id", "model_kwargs", and
                "trainer_kwargs" keys.

        Returns:
            FineTuningConfig object.
        """
        return cls(
            model_id=model_dict["model_id"],
            model_kwargs=model_dict["model_kwargs"],
            trainer_kwargs=hyperparameter_dict["trainer_kwargs"],
            early_stopping_kwargs=hyperparameter_dict.get(
                "early_stopping_kwargs", None
            ),
            peft_kwargs=hyperparameter_dict.get("peft_kwargs", None),
            add_padding_token=model_dict.get("add_padding_token", None),
        )

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "model_kwargs": self.model_kwargs,
            "trainer_kwargs": self.trainer_kwargs,
            "early_stopping_kwargs": self.early_stopping_kwargs,
        }
