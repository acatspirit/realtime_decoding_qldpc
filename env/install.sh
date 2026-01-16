#!/bin/bash
set -e

# load conda
conda env create -f env/environment.yml || true
conda activate realtime_decoding

pip install -e .

python -m ipykernel install --user --name realtime_decoding

if ! command -v cargo &> /dev/null; then
    echo "Rust not found. Please install Rust first."
    exit 1
fi

mkdir -p ~/external_repos
cd ~/external_repos

if [ ! -d relay ]; then
    git clone https://github.com/trmue/relay.git
fi

cd relay
pip install ".[stim]"

chmod +x env/install.sh