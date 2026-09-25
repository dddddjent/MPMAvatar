#!/usr/bin/env bash
# Template command, from the workspace root after source checkout: bash MPMAvatar/build_extensions.sh
set -eo pipefail
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
python -m pip install --no-build-isolation --no-deps \
  "$MPM_BUILD/pytorch3d" \
  "$MPM_BUILD/simple-knn" \
  "$MPM_BUILD/diff-gaussian-rasterization"
python -m pip check
