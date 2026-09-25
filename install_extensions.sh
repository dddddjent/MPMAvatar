#!/usr/bin/env bash
# Template command, from the workspace root after setup.sh's packages: bash MPMAvatar/install_extensions.sh
set -eo pipefail
MPM_REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
eval "$(conda shell.bash hook)"
conda activate mpmavatar
MPM_BUILD="$(dirname "$(dirname "$CONDA_PREFIX")")/build/mpmavatar"
export PIP_CACHE_DIR="$(dirname "$(dirname "$CONDA_PREFIX")")/pip-cache"
export CUDA_HOME="$CONDA_PREFIX"
export TORCH_CUDA_ARCH_LIST='8.0;8.6;9.0;12.0'
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/targets/x86_64-linux/include/cccl"
export LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib"
export CUB_HOME="$CONDA_PREFIX/targets/x86_64-linux/include/cccl"
export CC="$CXX"
export MAX_JOBS=6
export FORCE_CUDA=1
mkdir -p "$MPM_BUILD"
git clone --depth 1 https://github.com/facebookresearch/pytorch3d.git "$MPM_BUILD/pytorch3d"
git clone --depth 1 https://gitlab.inria.fr/bkerbl/simple-knn.git "$MPM_BUILD/simple-knn"
git clone --depth 1 --recursive https://github.com/slothfulxtx/diff-gaussian-rasterization.git \
  "$MPM_BUILD/diff-gaussian-rasterization"
sed -i '/#include <iostream>/a #include <cstdint>' \
  "$MPM_BUILD/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h"
bash "$MPM_REPO/build_extensions.sh"
