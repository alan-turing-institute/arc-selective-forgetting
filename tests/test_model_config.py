import arcsf.constants
from arcsf.models.config import ModelConfig


def test_model_config_from_yaml():
    model_dir = f"{arcsf.constants.MODEL_CONFIG_DIR}/dummy_gpt2"
    config_path = f"{model_dir}/dummy_gpt2.yaml"
    hyperparameters_path = f"{model_dir}/hyperparameters/dummy_hyperparams_config.yaml"
    model_cfg = ModelConfig.from_yaml(config_path, None, hyperparameters_path)
    assert model_cfg.model_id == "gpt2"
    assert model_cfg.model_kwargs["device_map"] == "auto"
    assert model_cfg.add_padding_token
    assert model_cfg.trainer_kwargs == {
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 5e-5,
        "num_train_epochs": 50,
        "logging_dir": "output/logs",
        "evaluation_strategy": "epoch",
        "logging_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "output_dir": "output",
    }
    assert model_cfg.peft_kwargs is None
    assert model_cfg.add_padding_token
    assert model_cfg.output_dir == "output"
