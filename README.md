# REALTIME_DECODING_QLDPC
Develop realtime decoding algorithms based on hard/soft decoding framework in arXiv:2510.25222 for qLDPC codes. Test performance with FPGA simulation. 


## Setup (local)
```bash
conda env remove -n realtime_decoding -y || true
conda env create -f env/environment.yml
conda activate realtime_decoding
pip install -e .
```
For the relay-bp package, install rust with the first line then do the following:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
mkdir -p ~/external_repos
cd ~/external_repos
git clone https://github.com/trmue/relay.git
cd relay
pip install ".[stim]"
```

for the ldpc-post-selection git

```bash
git clone --recurse-submodules https://github.com/seokhyung-lee/ldpc-post-selection.git
cd ldpc-post-selection
pip install -e .
# Optional: install codes used for the numerical analyses in our paper.
pip install -e ./simulations
```

To run the Union-Find decoding scripts, clone and install the decoder dependency outside this repository:
```bash
git clone [https://github.com/nbi-hyq/uf_decoder.git](https://github.com/nbi-hyq/uf_decoder.git)
cd uf_decoder
pip install -e .
### On the HPC only...
```bash
pip install cudaq
```
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
```


## Setup (DCC)
If you don't already have an account, set one up using this quick start guide: https://xc184.pages.oit.duke.edu/DCC/quick_start/. I recommend
using the SSH login option - you will need to generate an SSH key and add an agent. You can create a folder in the Brownlab directory (/hpc/group/brownlab) for the most memory access.

### Pull the repo
Navigate to your desired location (I run from Brownlab) and create a folder with your NetID. Pull the repo from Github or Gitlab (Github link: https://github.com/acatspirit/realtime_decoding_qldpc.git, GitLab link: https://gitlab.oit.duke.edu/am1155/realtime_decoding_qldpc.git). 

The workflow that I find most useful is continuously pushing from my local computer to Github or Gitlab, then pulling to the DCC version of the repo. You can then edit code and .sh executable scripts locally, and not have to use a text editor or some other complex software on the DCC. Then, you can write a script
or use mine to download from the DCC to your local computer. 

### Pull the Apptainer image
Download the realtime_decoding_qldpc.sif file to your local computer from the most recent GitLab commit. This will need to be downloaded each time we update
the environment or install new packages, since the full sync doesn't work on the DCC.

Once it is downloaded, scp the realtime_decoding_qldpc.sif file to the DCC realtime_decoding_qlpdc repo. This should live in the outermost folder. Now you are
good to run things on the DCC! This .sif image should create a local version of the exact environment and doesn't require any additional installs.

### Run code on the DCC
To get switching results, you will need to navigate to the simulation_scripts folder in the realtime_decoding_qldpc repo. From here, you can change the parameters and run the function get_ler_for_decoder_switching_dcc in ler_for_decoder_switching.py. Make sure to leave only this function uncommented. The submit_switching_ler.sh file runs teh ler_for_decoder_switching.py file in job batches. You will need to change some parameters in the submit_switching_ler.sh to reflect your local setup, and change the array size based on what you want to run. 

If you want to write your own DCC scripts, there are plenty of tutorials on the duke research computing website. To parallelize code, use arrays as outlined in the https://dcc.duke.edu/dcc/slurm/ tutorial. That is what we use in the submit_switching_ler.sh.