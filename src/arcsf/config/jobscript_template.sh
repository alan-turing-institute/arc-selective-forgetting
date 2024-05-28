#!/bin/bash
#SBATCH --account vjgo8416-sltv-forget
#SBATCH --qos turing
#SBATCH --job-name {{ job_name }}
#SBATCH --time {{ walltime }}
#SBATCH --nodes {{ node_number }}
#SBATCH --gpus {{ gpu_number }}
#SBATCH --output ./slurm_logs/{{ job_name }}-%j.out
#SBATCH --array=0-{{ array_number }}

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
echo "${SLURM_JOB_ID}: Job ${SLURM_ARRAY_TASK_ID} in the array"
python {{ script_name }} --experiment_name "{{ experiment_file }}_${SLURM_ARRAY_TASK_ID}"
