#!/bin/bash
#SBATCH --job-name=szlb-job
#SBATCH --partition=gpu-a100-Partition
#SBATCH -N 1
#SBATCH  --gres=gpu:1
#SBATCH --output=%j.out
#SBATCH --error=%j.err

python /lustre/home/kwchen/git/Stab2PPB/training/train_stab.py \
    --config config.json