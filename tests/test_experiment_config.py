import arcsf.constants
from arcsf.config.experiment import ExperimentConfig
from arcsf.data.config import DataConfig
from arcsf.models.config import ModelConfig


def test_data_config():
    path = f"{arcsf.constants.EXPERIMENT_CONFIG_DIR}/dummy_experiment_config.yaml"
    experiment_cfg = ExperimentConfig.from_yaml(path)
    assert isinstance(experiment_cfg.data_config, DataConfig)
    assert isinstance(experiment_cfg.model_config, ModelConfig)
