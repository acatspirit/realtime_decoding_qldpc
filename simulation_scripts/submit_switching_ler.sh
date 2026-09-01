#!/bin/bash
#SBATCH --job-name=qec_sim
#SBATCH --output=logs_%A_%a.out
#SBATCH --error=logs_%A_%a.err
#SBATCH --array=0-599 # should be len(ps) * len(code_names) * (num_shots // shots_per_job) - 1 , 6*20*5-1
#SBATCH --cpus-per-task=1
#SBATCH --partition=common,scavenger
#SBATCH --mem=10G
#SBATCH --time=8:00:00

# Navigate to your project directory (CHANGE THIS TO YOUR PROJECT DIRECTORY)
cd /hpc/group/brownlab/am1155/realtime_decoding_qldpc

# Unset host python variables to prevent them from bleeding into the container
unset PYTHONPATH

# Run using Apptainer, forcing the exact path to the container's internal Python (DO NOT CHANGE THIS PATH - should point to the .sif file and internal Python path)
apptainer exec realtime_decoding_qldpc.sif /opt/conda/envs/realtime_decoding/bin/python -u simulation_scripts/ler_for_decoder_switching.py