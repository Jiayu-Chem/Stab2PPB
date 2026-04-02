#!/bin/bash
#SBATCH --job-name=szlb-job
#SBATCH --partition=gpu-a100-Partition,gpu-2
#SBATCH -N 1
#SBATCH  --gres=gpu:1
#SBATCH --output=%j.out
#SBATCH --error=%j.err

python /lustre/home/kwchen/git/Stab2PPB/ppb/test_ppb.py \
    --config config.json \
    --use_wandb \
    --csv "/lustre/home/kwchen/dataset/PPB-Affinity/benchmark.csv" \
    --out_csv "ppb_test_results.csv" \
    --weight "best_stability_model.pt" \
    --run_name $1
