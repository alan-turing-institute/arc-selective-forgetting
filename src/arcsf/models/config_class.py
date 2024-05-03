from arcsf.config.config_class import Config


class ModelConfig(Config):

    def __init__(
        self,
        model_id: str,
        model_kwargs: dict,
        trainer_kwargs: dict,
        use_wandb: bool,
    ) -> None:
        super().__init__()

        # Main model kwargs
        self.model_id = model_id
        self.model_kwargs = model_kwargs

        # Output dir
        self.output_dir = trainer_kwargs["output_dir"]

        # Training arguments: batch size
        self.train_batch_size = trainer_kwargs["train_batch_size"]
        self.eval_batch_size = trainer_kwargs["eval_batch_size"]
        self.eval_accumulation_steps = trainer_kwargs["eval_accumulation_steps"]

        # Training arguments: hyperparams
        self.learning_rate = trainer_kwargs["learning_rate"]
        if isinstance(self.learning_rate, str):
            self.learning_rate = float(self.learning_rate)
        self.num_train_epochs = trainer_kwargs["num_train_epochs"]

        # Evaluation
        self.evaluation_strategy = trainer_kwargs["evaluation_strategy"]

        # Logging
        # TODO: work out how to manage this in line w/ varying dataset size
        self.logging_strategy = trainer_kwargs["logging_strategy"]
        self.logging_dir = f'{trainer_kwargs["output_dir"]}/logs'
        self.logging_steps = trainer_kwargs["logging_steps"]

        # Wandb
        self.use_wandb = use_wandb

        # Early stopping - TODO make optional
        self.save_strategy = trainer_kwargs["save_strategy"]
        self.save_steps = trainer_kwargs["logging_steps"]
        self.load_best_model_at_end = trainer_kwargs["load_best_model_at_end"]
        self.metric_for_best_model = trainer_kwargs["metric_for_best_model"]

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
            use_wandb=dict["use_wandb"],
        )

    # TODO: define to_dict method
    def to_dict(self) -> dict:
        return {"todo": "not done"}
