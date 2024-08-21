import wandb

sweep_config = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "eval/retain_mean_rougeL_recall"},
    "parameters": {
        "learning_rate": {"values": [5e-4, 1e-4, 5e-5, 1e-5]},
        "num_train_epochs": {"values": [5, 25, 50]},
        "weight_decay": {"values": [0, 0.01]},
        "warmup_ratio": {"values": [0, 0.04]},
        "lr_scheduler_type": {"values": ["linear", "cosine"]},
        "per_device_train_batch_size": {"values": [4, 8, 16, 32]},
        "experiment_config": {"values": ["experiment_2_relationships_gpt2/retain_0"]},
    },
}

sweep_id = wandb.sweep(
    sweep=sweep_config, project="selective-forgetting", entity="turing-arc"
)
print("Started sweep", sweep_id)
