#!/bin/bash
#SBATCH --job-name=qec_sim
#SBATCH --output=logs_%A_%a.out
#SBATCH --error=logs_%A_%a.err
#SBATCH --array=0-299
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=8:00:00

# Navigate to your project directory
cd /hpc/group/brownlab/am1155/realtime_decoding_qldpc

# Run using Apptainer, passing the array task ID to your script
apptainer exec realtime_decoding_qldpc.sif python simulation_scripts/ler_for_decoder_switching.py