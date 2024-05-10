from arcsf.config.config import Config


class ModelConfig(Config):

    def __init__(
        self,
        model_id: str,
        model_kwargs: dict,
        trainer_kwargs: dict,
        early_stopping_kwargs: dict | None,
    ) -> None:
        super().__init__()

        # Process main inputs
        self.model_id = model_id
        self.model_kwargs = model_kwargs
        self.early_stopping_kwargs = early_stopping_kwargs

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
    def from_dict(cls, dict) -> "ModelConfig":
        """Create a FineTuningConfig from a config dict.

        Args:
            config: Dict that must contain "model_id", "model_kwargs", and
                "trainer_kwargs" keys.

        Returns:
            FineTuningConfig object.
        """
        return cls(
            model_id=dict["model_id"],
            model_kwargs=dict["model_kwargs"],
            trainer_kwargs=dict["trainer_kwargs"],
            early_stopping_kwargs=dict.get("early_stopping_kwargs", None),
        )

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "model_kwargs": self.model_kwargs,
            "trainer_kwargs": self.trainer_kwargs,
            "early_stopping_kwargs": self.early_stopping_kwargs,
        }
