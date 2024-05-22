from arcsf.config.config import Config


class DataConfig(Config):
    """Data config class.

    Attributes:
        dataset_name: Name of dataset to use, corresponding to entry in data_dict in
                      arcsf.data.data_module.data_dict
        data_kwargs: Dict of kwargs used to load the dataset
    """

    def __init__(
        self,
        dataset_name: str,
        data_kwargs: dict,
    ) -> None:
        super().__init__()

        # Process main inputs
        self.dataset_name = (dataset_name,)
        self.data_kwargs = data_kwargs

    @classmethod
    def from_dict(cls, data_dict) -> "DataConfig":
        """Create a FineTuningConfig from a config dict.

        Args:
            config: Dict that must contain "model_id", "model_kwargs", and
                "trainer_kwargs" keys.

        Returns:
            FineTuningConfig object.
        """
        return cls(**data_dict)

    def to_dict(self) -> dict:
        return {
            "data_kwargs": self.data_kwargs,
        }
