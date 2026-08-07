#!/bin/bash
#SBATCH --job-name=decoder_switching          # Name of your job
#SBATCH --output=logs/sim_%A_%a.out           # Standard output log (%A = job ID, %a = task ID)
#SBATCH --error=logs/sim_%A_%a.err            # Standard error log
#SBATCH --partition=common                    # Use the standard DCC partition
#SBATCH --time=07:59:00                       # Max run time (keeps it under the 8-hour limit)
#SBATCH --mem=4G                              # Memory per task (4GB is usually plenty for a single 1-core QEC chunk)
#SBATCH --cpus-per-task=1                     # 1 core per job (Slurm handles the parallelization now)
#SBATCH --array=0-299                         # num codes x num probabilities x total shots / batch shots - 1

# Print some helpful debugging info to the log
echo "Starting job $SLURM_ARRAY_JOB_ID, task $SLURM_ARRAY_TASK_ID"
echo "Running on node: $(hostname)"


# # 2. Activate your environment
# conda activate realtime_decoding

# 3. Create the logs directory if it doesn't exist yet
mkdir -p logs

# 4. Run the python script
# (Make sure this points to the file where you saved the refactored function!)
apptainer exec /hpc/group/brownlab/am1155/realtime_decoding_qldpc/realtime_decoding_qldpc.sif python ler_for_decoder_switching.py

echo "Task $SLURM_ARRAY_TASK_ID completed."