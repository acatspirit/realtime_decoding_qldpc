# realtime_decoding_qldpc
Develop realtime decoding algorithms based on hard/soft decoding framework in arXiv:2510.25222 for qLDPC codes. Test performance with FPGA simulation. 


## Setup (HPC / local)
```bash
conda env remove -n realtime-decoding -y || true
conda env create -f env/environment.yml
conda activate realtime-decoding
pip install -e .