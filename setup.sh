#!/usr/bin/env bash
# Template command, from the workspace root: bash MPMAvatar/setup.sh
# Build CUDA extensions for A100, A40, H200, and RTX 5080.
set -eo pipefail
MPM_REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
eval "$(conda shell.bash hook)"
conda create -n mpmavatar python=3.10 pip -y
conda activate mpmavatar
export PIP_CACHE_DIR="$(dirname "$(dirname "$CONDA_PREFIX")")/pip-cache"
conda install -c conda-forge -y \
  cuda-nvcc=13.0 cuda-cudart-dev=13.0 cuda-cccl=13.0 \
  libcublas-dev=13.1 libcusparse-dev=12.6 libcusolver-dev=12.0 \
  gcc_linux-64=13 gxx_linux-64=13 ninja cmake ffmpeg decord=0.6.0 numpy=1.25.0
conda deactivate
conda activate mpmavatar
conda env config vars set -n mpmavatar CUDA_HOME="$CONDA_PREFIX" TORCH_CUDA_ARCH_LIST='8.0;8.6;9.0;12.0'
python -m pip install setuptools==75.8.0 wheel
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r "$MPM_REPO/requirements.txt"
bash "$MPM_REPO/install_extensions.sh"
