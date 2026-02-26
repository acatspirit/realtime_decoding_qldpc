import os
import sys
import numpy as np
import pandas as pd

from realtime_decoding.decoding import (
    get_log_error_CL_BP_MWPM,
    get_log_error_CL_MWPM,
)

# -------------------------
# Parameters
# -------------------------

num_shots = 1
max_iter = 30
t = 0.9
memory_type = "X"

d_list = [5, 7, 9, 13]
p_list = np.linspace(0.005, 0.013, 10)

# -------------------------
# Slurm array index logic
# -------------------------

task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

num_p = len(p_list)
num_d = len(d_list)

total_jobs = num_d * num_p

if task_id >= total_jobs:
    raise ValueError(f"Task ID {task_id} exceeds total jobs {total_jobs}")

d_index = task_id // num_p
p_index = task_id % num_p

d = d_list[d_index]
p = p_list[p_index]

print(f"Running job for d={d}, p={p}")

# -------------------------
# Run simulations
# -------------------------

bp_mwpm_log_err = get_log_error_CL_BP_MWPM(
    d=d,
    p=p,
    shots=num_shots,
    max_iter=max_iter,
    memory_type=memory_type,
    t=t,
)

mwpm_log_err = get_log_error_CL_MWPM(
    d=d,
    p=p,
    memory_type=memory_type,
    shots=num_shots,
)

# -------------------------
# Save results
# -------------------------

row = {
    "d": d,
    "p": p,
    "num_shots": num_shots,
    "max_iter": max_iter,
    "t": t,
    "memory_type":memory_type,
    "MWPM_log_error": mwpm_log_err,
    "BP_MWPM_log_error": bp_mwpm_log_err,
}

# Optional metadata (recommended)
row["slurm_job_id"] = os.environ.get("SLURM_JOB_ID", "local")
row["slurm_array_id"] = os.environ.get("SLURM_ARRAY_TASK_ID", "local")

# -------------------------
# Create results directory
# -------------------------

results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Clean filename (avoid floating point weirdness)
p_str = f"{p:.6f}"

filename = f"result_d{d}_p{p_str}.csv"
output_file = os.path.join(results_dir, filename)

# -------------------------
# Write file
# -------------------------

df = pd.DataFrame([row])
df.to_csv(output_file, index=False)

# print(f"Saved result to {output_file}")