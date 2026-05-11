#!/bin/bash
#SBATCH --job-name=szlb-job
#SBATCH --partition=gpu-a100-Partition
#SBATCH -N 1
#SBATCH  --gres=gpu:1
#SBATCH --output=%j.out
#SBATCH --error=%j.err

# export WANDB_MODE=offline
# export TMPDIR='/lustre/home/kwchen/tmp'
# export WANDB_DIR='/lustre/home/kwchen/tmp'

python /lustre/home/kwchen/git/Stab2PPB/stab2ppb/train.py \
    --config config.json \
    --use_wandb