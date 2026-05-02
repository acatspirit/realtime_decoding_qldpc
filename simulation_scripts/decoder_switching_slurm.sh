#!/bin/bash
#SBATCH --job-name=qec_sw
#SBATCH --array=0-149
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%a.out

# Activate your environment
# conda activate realtime_decoding
python decoder_switching.py