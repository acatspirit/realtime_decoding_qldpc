#!/bin/bash
#SBATCH --job-name=qec_sim
#SBATCH --array=0-6332%500
#SBATCH --mem=15G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%a.out

# Activate your environment
# conda activate realtime_decoding
python decoder_switching.py