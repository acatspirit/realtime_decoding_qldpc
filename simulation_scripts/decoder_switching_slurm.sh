#!/bin/bash
#SBATCH --job-name=qec_sim
#SBATCH --array=3001-4000 # total 6332
#SBATCH --mem=15G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%a.out

# Activate your environment
# conda activate realtime_decoding
python decoder_switching.py