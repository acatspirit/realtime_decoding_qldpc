#!/bin/bash
#SBATCH --job-name=bb_switching
#SBATCH --array=0-1000 
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%a.out

# Activate your environment
# conda activate realtime_decoding
python decoder_switching.py