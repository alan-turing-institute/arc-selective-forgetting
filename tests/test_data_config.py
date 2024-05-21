import arcsf.constants
from arcsf.data.config import DataConfig

# from arcsf.constants import DATA_CONFIG_DIR


def test_data_config_from_yaml():
    path = f"{arcsf.constants.DATA_CONFIG_DIR}/dummy_data_config.yaml"
    data_cfg = DataConfig.from_yaml(path)
    assert data_cfg.data_kwargs["granularity"] == "question"
    assert data_cfg.data_kwargs["stratified"]
    assert data_cfg.data_kwargs["forget_random"]
    assert data_cfg.data_kwargs["forgotten_author_fraction"] == 0.2
    assert data_cfg.data_kwargs["forgotten_fact_fraction"] == 0.2
