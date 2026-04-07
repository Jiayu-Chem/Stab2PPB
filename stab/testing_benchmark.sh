#!/bin/bash
#SBATCH --job-name=szlb-job
#SBATCH --partition=gpu-a100-Partition,gpu-2
#SBATCH -N 1
#SBATCH  --gres=gpu:1
#SBATCH --output=benchmark.out
#SBATCH --error=benchmark.err

python /lustre/home/kwchen/git/Stab2PPB/stab/test_benchmark.py \
    --config ./config.json \
    --weights ./best_stability_model.pt \
    --bench_json /lustre/home/kwchen/git/Stab2PPB/bench_path.json