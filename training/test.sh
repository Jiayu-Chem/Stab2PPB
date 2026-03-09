#!/bin/bash
#SBATCH --job-name=szlb-job
#SBATCH --partition=gpu-2
#SBATCH -N 1
#SBATCH  --gres=gpu:1
#SBATCH --output=test_%j.out
#SBATCH --error=test_%j.err

python /lustre/home/kwchen/git/Stab2PPB/training/test_stab.py \
    --config config.json \
    -p best_stability_model.pt
