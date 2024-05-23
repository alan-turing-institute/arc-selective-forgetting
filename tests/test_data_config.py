import arcsf.constants
from arcsf.data.config import DataConfig


def test_data_config_from_yaml():
    path = f"{arcsf.constants.DATA_CONFIG_DIR}/dummy_data_config.yaml"
    data_cfg = DataConfig.from_yaml(path)
    assert data_cfg.dataset_name == "tofu"
    assert data_cfg.data_kwargs == {
        "granularity": "question",
        "stratified": True,
        "forget_random": True,
        "forgotten_author_fraction": 0.2,
        "forgotten_fact_fraction": 0.2,
    }
