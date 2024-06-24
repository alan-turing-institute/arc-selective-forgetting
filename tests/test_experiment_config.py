import arcsf.constants
from arcsf.config.experiment import ExperimentConfig
from arcsf.data.config import DataConfig
from arcsf.models.config import ModelConfig


def test_experiment_config_from_yaml():
    path = f"{arcsf.constants.EXPERIMENT_CONFIG_DIR}/dummy_experiment_config.yaml"
    experiment_cfg = ExperimentConfig.from_yaml(path)
    assert isinstance(experiment_cfg.data_config, DataConfig)
    assert isinstance(experiment_cfg.model_config, ModelConfig)
    assert experiment_cfg.train_type == "retain"
    assert experiment_cfg.full_model_name == "this_string_not_used_yet"
    assert experiment_cfg.use_wandb
    assert experiment_cfg.wandb_config == {
        "entity": "turing-arc",
        "project": "selective-forgetting",
        "log_model": "false",
        "group": "test",
    }
    assert experiment_cfg.run_name == "dummy_gpt2-dummy_data_config-retain-42"
