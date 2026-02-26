#!/bin/bash
#SBATCH --job-name=threshold_sweep
#SBATCH --output=logs/threshold_%A_%a.out
#SBATCH --error=logs/threshold_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-39

# 4 d values × 10 p values = 40 jobs

python simulation_data.py