#!/bin/bash
#SBATCH --job-name=szlb-job
#SBATCH --partition=gpu-a100-Partition,gpu-2
#SBATCH -N 1
#SBATCH  --gres=gpu:1
#SBATCH --output=test_ppi.out
#SBATCH --error=test_ppi.err

python /lustre/home/kwchen/git/Stab2PPB/ppb/test_ppi.py \
    --config ./config.json \
    --model_path ./best_stability_model.pt \
    --csv_file /lustre/home/kwchen/dataset/3766/final_dataset_clean.csv \
    --pdb_dir /lustre/home/kwchen/dataset/3766/structure/ \
    --output ppi_zeroshot_results.csv \
    --batch_size 16