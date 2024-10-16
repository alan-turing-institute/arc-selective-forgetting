import yaml

from arcsf.config.config import Config


class ModelConfig(Config):
    """Model config class.

    Attributes:
        model_id: HuggingFace ID for the model
        model_kwargs: Dict of kwargs relating to the model itself, including
                      hyperparameters
        trainer_kwargs: Dict of kwargs passed to the trainer
        early_stopping_kwargs: Optional dict of kwargs relating to early stopping
        peft_kwargs: Optional dict of LoRA kwargs if using LoRA fine-tuning
        qa_formatter_kwargs: Dict containing question_template and answer_template,
            see QAFormatter class for details.
        add_padding_token: Optional argument which if True means that if no padding
                           token is present in the tokenizer it will be added.
        output_dir: Relative path for storing outputs produced during training. In
                    training script, subdirectory of save dir.
    """

    def __init__(
        self,
        model_id: str,
        model_kwargs: dict,
        trainer_kwargs: dict,
        early_stopping_kwargs: dict | None,
        peft_kwargs: dict | None,
        qa_formatter_kwargs: dict,
        add_padding_token: bool = False,
    ) -> None:
        super().__init__()

        # Process main inputs
        self.model_id = model_id
        self.model_kwargs = model_kwargs
        self.peft_kwargs = peft_kwargs
        self.early_stopping_kwargs = early_stopping_kwargs
        self.add_padding_token = add_padding_token
        self.qa_formatter_kwargs = qa_formatter_kwargs

        # Process trainer kwargs
        if isinstance(trainer_kwargs["learning_rate"], str):
            trainer_kwargs["learning_rate"] = float(trainer_kwargs["learning_rate"])
        trainer_kwargs["logging_dir"] = f'{trainer_kwargs["output_dir"]}/logs'
        if "logging_steps" in trainer_kwargs.keys():
            # Keep save steps and eval steps in line with logging steps
            trainer_kwargs["save_steps"] = trainer_kwargs["logging_steps"]
            trainer_kwargs["eval_steps"] = trainer_kwargs["logging_steps"]

        # Add trainer kwargs to self
        self.trainer_kwargs = trainer_kwargs

        # Output dir
        self.output_dir = trainer_kwargs["output_dir"]

    @classmethod
    def from_yaml(
        cls, config_path: str, model_path: str | None, hyperparameter_path: str
    ) -> "ModelConfig":
        """Create a ModelConfig from model and hyperparameter yaml files.

        Args:
            model_config: Path to yaml file from which model kwargs can be read.
            model_path: HuggingFace model ID or local path to model (if None defaults
                to model_id in model_config)
            hyperparameter_path: Path to yaml file from which hyperparameters can be
                                 read.

        Returns:
            ModelConfig object.
        """
        with open(config_path, "r") as f:
            model_config = yaml.safe_load(f)
        with open(hyperparameter_path, "r") as f:
            hyperparameter_config = yaml.safe_load(f)
        return cls.from_dict(model_config, model_path, hyperparameter_config)

    @classmethod
    def from_dict(cls, model_dict, model_path, hyperparameter_dict) -> "ModelConfig":
        """Create a ModelConfig from model and hyperparameter config dicts.

        Args:
            model_dict: Dict contaiing model-specific arguments.
            hyperparameter_dict: Dict containing hyperparameters.

        Returns:
            ModelConfig object.
        """
        return cls(
            model_id=model_path or model_dict["model_id"],
            model_kwargs=model_dict["model_kwargs"],
            trainer_kwargs=hyperparameter_dict["trainer_kwargs"],
            early_stopping_kwargs=hyperparameter_dict.get(
                "early_stopping_kwargs", None
            ),
            peft_kwargs=hyperparameter_dict.get("peft_kwargs", None),
            qa_formatter_kwargs=model_dict["qa_formatter_kwargs"],
            add_padding_token=model_dict.get("add_padding_token", None),
        )

    def to_dict(self) -> dict:
        """
        Converts the ModelConfig's attributes into a dict.

        Returns:
            dict: Dict containing the ModelConfig's attributes.
        """
        return {
            "model_id": self.model_id,
            "model_kwargs": self.model_kwargs,
            "trainer_kwargs": self.trainer_kwargs,
            "peft_kwargs": self.peft_kwargs,
            "early_stopping_kwargs": self.early_stopping_kwargs,
            "qa_formatter_kwargs": self.qa_formatter_kwargs,
            "add_padding_token": self.add_padding_token,
        }
