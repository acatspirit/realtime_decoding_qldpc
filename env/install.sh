#!/bin/bash
set -e

# Load conda if needed (cluster-specific)
# module load anaconda3

conda env create -f env/environment.yml || true
conda activate realtime_decoding

pip install -e .

python -m ipykernel install --user --name realtime_decoding

chmod +x env/install.sh