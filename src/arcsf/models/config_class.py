from arcsf.config.config_class import Config


class ModelConfig(Config):

    def __init__(
        self,
        model_id: str,
        model_kwargs: dict,
        trainer_kwargs: dict,
    ) -> None:
        super().__init__()

        # Main model kwargs
        self.model_id = model_id
        self.model_kwargs = model_kwargs

        # Process trainer kwargs
        if isinstance(trainer_kwargs["learning_rate"], str):
            trainer_kwargs["learning_rate"] = float(trainer_kwargs["learning_rate"])
        trainer_kwargs["logging_dir"] = f'{trainer_kwargs["output_dir"]}/logs'
        if "logging_steps" in trainer_kwargs.keys():
            trainer_kwargs["save_steps"] = trainer_kwargs["logging_steps"]

        # TODO: work out how to manage logging steps in line w/ varying dataset size

        # Add trainer kwargs to self
        self.trainer_kwargs = trainer_kwargs

        # Output dir
        self.output_dir = trainer_kwargs["output_dir"]

        # TODO make early stopping optional

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
        )

    # TODO: define to_dict method
    def to_dict(self) -> dict:
        return {"todo": "not done"}
