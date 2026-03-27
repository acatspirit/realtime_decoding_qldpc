#!/bin/bash
#SBATCH --job-name=threshold_sweep
#SBATCH --output=logs/threshold_%A_%a.out
#SBATCH --error=logs/threshold_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --mem=6G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-39

# 4 d values × 10 p values = 40 jobs

python -u simulation_data.py