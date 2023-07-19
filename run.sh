#!/bin/bash
#SBATCH -n10
#SBATCH --job-name=train
#SBATCH -N1
#SBATCH -p DGX
#SBATCH --gpus=1
#SBATCH --mem=50gb
#SBATCH --time=48:00:00
#SBATCH --output=my_job_%j.out
#SBATCH --error=my_job_%j.err


#Comment 
conda activate torch
echo $SLURM_JOB_ID
# Run the application
python -m src_code.script.dqn --output_log_dir ./results/cnn/${SLURM_JOB_ID} --output_checkpoint_dir ./results/cnn/${SLURM_JOB_ID}/checkpoints