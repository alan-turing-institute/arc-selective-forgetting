from abc import ABC, abstractmethod

import yaml


class Config(ABC):

    def __init__(
        self,
    ) -> None:
        super().__init__()

    @classmethod
    @abstractmethod
    def from_dict(cls, dict) -> "Config":
        """Create a FineTuningConfig from a config dict.

        Args:
            config: Dict that must contain "model_name" and "dataset_name" keys. Can
                also contain "run_name", "random_state", "dataset_args",
                "training_args", "use_wandb" and "wandb_args" keys. If "use_wandb" is
                not specified, it is set to True if "wandb" is in the config dict.

        Returns:
            FineTuningConfig object.
        """
        raise NotImplementedError

    @classmethod
    def read_yaml(cls, path: str) -> "Config":
        """Create a FineTuningConfig from a yaml file.

        Args:
            path: Path to yaml file.

        Returns:
            FineTuningConfig object.
        """
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls.from_dict(config=config)

    @abstractmethod
    def to_dict(self) -> dict:
        """Convert the config to a dict.

        Returns:
            Dict representation of the config.
        """
        raise NotImplementedError
