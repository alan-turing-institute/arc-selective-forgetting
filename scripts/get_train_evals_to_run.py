"""
Get paths to eval output files.
"""

import os
from glob import glob

output_dir = "output"
include = [
    "experiment_1_granularity_gpt2",
    "experiment_1_granularity_gpt2_longer_train",
    "experiment_1_granularity_gpt2_one_each",
    "experiment_1_granularity_gpt2_remaining",
    "experiment_1_granularity_llama",
    "experiment_1_granularity_llama_seeds",
    "experiment_1_granularity_phi",
    "experiment_1_granularity_phi_seeds",
    "experiment_2_relationships_gpt2",
    "experiment_2_relationships_llama",
    "experiment_2_relationships_phi",
]

eval_output_paths = sum(
    (
        glob(f"{output_dir}/{experiment}/**/eval_outputs.json", recursive=True)
        for experiment in include
    ),
    [],
)

# directories with saved models (have eval_outputs.json in parent output directory,
# indicates a job that finished successfully)
model_dirs = [
    path.removesuffix("/eval_outputs.json")
    for path in eval_output_paths
    if "/eval_outputs/" not in path  # exclude eval_outputs in re-run subdirectories
]

# models without train set eval outputs
no_train_eval = [
    path
    for path in model_dirs
    if len(glob(path + "/eval_outputs/**/train_set_eval_outputs.json")) == 0
]
print(
    f"Found {len(model_dirs)} models, of which {len(no_train_eval)} have no train set "
    "eval outputs."
)

os.makedirs("train_scripts/train_eval_jobs", exist_ok=True)
with open("train_scripts/train_eval_jobs/train_eval_jobs.txt", "w") as f:
    f.write("\n".join(no_train_eval))
