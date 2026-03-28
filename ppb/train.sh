#!/bin/bash
#SBATCH --job-name=szlb-job
#SBATCH --partition=gpu-a100-Partition
#SBATCH -N 1
#SBATCH  --gres=gpu:1
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# export WANDB_MODE=offline

python /lustre/home/kwchen/git/Stab2PPB/ppb/train_ppb.py \
    --config config.json \
    --fold -1 \
    --use_wandb