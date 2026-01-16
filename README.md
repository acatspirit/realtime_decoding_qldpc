# REALTIME_DECODING_QLDPC
Develop realtime decoding algorithms based on hard/soft decoding framework in arXiv:2510.25222 for qLDPC codes. Test performance with FPGA simulation. 


## Setup (HPC / local)

conda env remove -n realtime_decoding -y || true
conda env create -f env/environment.yml
conda activate realtime_decoding
pip install -e .


### What goes where (important)

#### `src/realtime_decoding/`
**Authoritative, paper-quality code.**

- Anything used to generate results for a paper **must** live here
- Nothing in `src/` should depend on notebooks.

---

#### `notebooks/`
**Exploratory and analysis notebooks only.**

- Notebooks are for scratch code
- No final results in notebooks please :D

Example:
```python
from realtime_decoding.decoding import my_decoder