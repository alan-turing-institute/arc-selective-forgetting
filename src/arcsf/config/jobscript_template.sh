#!/bin/bash
#SBATCH --account vjgo8416-locomoset
#SBATCH --qos turing
#SBATCH --job-name {{ job_name }}
#SBATCH --time {{ walltime }}
#SBATCH --nodes {{ node_number }}
#SBATCH --gpus {{ gpu_number }}
#SBATCH --cpus-per-gpu {{ cpu_per_gpu }}
#SBATCH --array=1-{{ array_number }}
#SBATCH --output ./slurm_logs/locomoset_{{ config_type }}_experiment-%j.out

# Load required modules here (pip etc.)
module purge
module load baskerville
module load Miniconda3/4.10.3
export CONDA_PKGS_DIRS=/tmp
eval "$(${EBROOTMINICONDA3}/bin/conda shell.bash hook)"

# This should be edited by the setup script to point in the
# right direction for the configs and run call
CONFIGSPATH={{ config_path }}/config_{{ config_type }}_${SLURM_ARRAY_TASK_ID}.yaml
RUN_CALL=locomoset_run_{{ config_type }}

# Define the path to your Conda environment (modify as appropriate)
CONDA_ENV_PATH=/bask/projects/v/vjgo8416-locomoset/ARC-LoCoMoSeT/locomosetenv
conda activate ${CONDA_ENV_PATH}

# Run script
echo 'Starting task with id' ${SLURM_ARRAY_TASK_ID}
$RUN_CALL ${CONFIGSPATH}
