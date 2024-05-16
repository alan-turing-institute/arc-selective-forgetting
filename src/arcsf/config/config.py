from abc import ABC, abstractmethod

import yaml


class Config(ABC):
    """Base class for config files."""

    def __init__(
        self,
    ) -> None:
        super().__init__()

    @classmethod
    @abstractmethod
    def from_dict(cls, dict) -> "Config":
        """Create a Config from a config dict.

        Args:
            config: Dict containing arguments with which to initialise the Config.

        Returns:
            Config object.
        """
        raise NotImplementedError

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Create a Config from a yaml file.

        Args:
            path: Path to yaml file from which a config dict can be read.

        Returns:
            Config object.
        """
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls.from_dict(config)

    @abstractmethod
    def to_dict(self) -> dict:
        """Convert the config to a dict.

        Returns:
            Dict representation of the config.
        """
        raise NotImplementedError
