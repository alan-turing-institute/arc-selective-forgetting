#!/bin/bash
#SBATCH --account vjgo8416-sltv-forget
#SBATCH --qos turing
#SBATCH --job-name example-job
#SBATCH --time 3-0:0:0
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --cpus-per-gpu 36
#SBATCH --output ./slurm_logs/example_experiment-%j.out

# Load required modules here (pip etc.)
module purge
module load baskerville
module load Miniconda3/4.10.3
export CONDA_PKGS_DIRS=/tmp
eval "$(${EBROOTMINICONDA3}/bin/conda shell.bash hook)"

# Define the path to your Conda environment (modify as appropriate)
CONDA_ENV_PATH=/bask/projects/v/vjgo8416-sltv-forget/sfenv
conda activate ${CONDA_ENV_PATH}

# Run script
python scripts/train.py --experiment_name example_experiment_config
