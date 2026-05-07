#!/bin/bash
#SBATCH --job-name=bb_switching
#SBATCH --array=0-31664 # total 6332
#SBATCH --mem=20G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%a.out

# Activate your environment
# conda activate realtime_decoding
python decoder_switching.py