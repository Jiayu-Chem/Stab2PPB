#!/bin/bash
#SBATCH --job-name=szlb-job
#SBATCH --partition=gpu-a100-Partition,gpu-2
#SBATCH -N 1
#SBATCH  --gres=gpu:1
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# export WANDB_MODE=offline

python /lustre/home/kwchen/git/Stab2PPB/stab/train_stab.py \
    --config config.json \
    --use_wandb