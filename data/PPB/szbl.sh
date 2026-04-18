#!/bin/bash
#SBATCH --job-name=szlb-job
#SBATCH --partition=cu-1,cpuPartition
#SBATCH -N 4
#SBATCH --ntasks-per-node=16
#SBATCH --output=%j.out
#SBATCH --error=%j.err

python preprocess_data.py