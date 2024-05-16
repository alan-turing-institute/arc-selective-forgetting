from arcsf.config.config import Config


class DataConfig(Config):

    def __init__(
        self,
        data_kwargs: dict,
    ) -> None:
        super().__init__()

        # Process main inputs
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
        return cls(
            data_kwargs=data_dict,
        )

    def to_dict(self) -> dict:
        return {
            "data_kwargs": self.data_kwargs,
        }
