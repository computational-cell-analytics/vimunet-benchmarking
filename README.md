# Benchmarking Experiments for "ViM-UNet: Vision Mamba for Biomedical Segmentation"

Experiments are performed on:
- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)
- [U-Mamba](https://github.com/bowang-lab/U-Mamba)

## Installation

### For nnU-Net:

[Here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md) is the detailed instruction on how to install nnU-Net on your system.

TLDR:

1. Install [PyTorch](https://pytorch.org/get-started/locally/)
2. Install nnU-Net from source:
```bash
$ git clone https://github.com/MIC-DKFZ/nnUNet.git
$ cd nnUNet
$ pip install -e .
```

### For U-Mamba:
1. Create a new mamba environment:
```bash
$ mamba env create -n umamba python=3.10 -y
$ mamba activate umamba
```

2. Install `PyTorch`:
```
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

3. Install packaging: `pip install packaging`

> `CUDA_HOME` needs to match the installed cuda version, and the path should be visible. For HLRN users, here's an example: `export CUDA_HOME=/usr/local/cuda-11.8/`.

4. Install `causal-conv1d`: `pip install causal-conv1d==1.1.1`
8. Install Mamba: `pip install mamba-ssm`
9. Clone the repository from scratch and install U-Mamba (we store the data at `U-Mamba/data` for performing the experiments)
```
$ git clone https://github.com/bowang-lab/U-Mamba.git
$ cd U-Mamba/umamba
$ pip install -e .
```
