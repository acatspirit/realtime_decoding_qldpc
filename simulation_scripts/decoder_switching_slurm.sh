#!/bin/bash
#SBATCH --job-name=qec_sim
#SBATCH --array=0-662
#SBATCH --cpus-per-task=1
#SBATCH --mem=15G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%a.out

# Activate your environment
# conda activate realtime_decoding
python decoder_switching.py