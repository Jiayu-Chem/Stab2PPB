#!/bin/bash
#SBATCH --job-name=szlb-job
#SBATCH --partition=gpu-a100-Partition,gpu-2
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --output=%A_%a.out
#SBATCH --error=%A_%a.err

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python /lustre/home/kwchen/git/Stab2PPB/ppb/train_ppb.py \
    --config config.json \
    --fold $SLURM_ARRAY_TASK_ID \
    --use_wandb