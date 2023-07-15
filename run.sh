#!/bin/bash
#SBATCH -n12
#SBATCH --job-name=train
#SBATCH -N1
#SBATCH -p DGX
#SBATCH --gpus=1
#SBATCH --mem=200gb
#SBATCH --time=48:00:00
#SBATCH --output=my_job_%j.out
#SBATCH --error=my_job_%j.err


#Comment 
conda activate torch
echo $SLURM_JOB_ID
# Run the application
python -m src_code.script.dqn_new --output_log_dir ./results/${SLURM_JOB_ID} --output_checkpoint_dir ./checkpoint/${SLURM_JOB_ID}