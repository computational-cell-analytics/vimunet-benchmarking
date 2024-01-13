# UMamba-Experiments
Experiments on "U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation"

Installation:
1. `mamba env create -n umamba python=3.10 -y`
2. `mamba activate umamba`
3. PyTorch installation: `mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`
4. `pip install packaging`
5. (OPTIONAL FOR HLRN USERS) `export CUDA_HOME=/usr/local/cuda-11.8/` (for spotting `nvcc`)
6. `pip install causal-conv1d==1.1.1`
7. `pip install mamba-ssm`
8. Clone the repo to scratch (as we will store the data inside the repo) - `git clone https://github.com/bowang-lab/U-Mamba.git`
9. `cd U-Mamba/umamba` -> `pip install -e .`
