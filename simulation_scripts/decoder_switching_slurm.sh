#!/bin/bash
#SBATCH --job-name=bb_uf
#SBATCH --array=0-9999 
#SBATCH --mem=2G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%a.out

# Activate your environment
# conda activate realtime_decoding
python decoder_switching.py