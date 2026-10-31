#!/usr/bin/env bash
# Week 5: build the two model environments on a Colab VM (they need different torch builds, so two envs).
#
#   bash install_envs.sh micromamba
#   bash install_envs.sh trellis2          # ~20-40 min, compiles CUDA extensions (needs a GPU runtime: setup.sh calls nvidia-smi)
#   bash install_envs.sh hunyuan           # ~15-30 min
#   bash install_envs.sh pack   <dir>      # tar the envs + repos to Drive so the A100 session can skip the install
#   bash install_envs.sh unpack <dir>
#
# Everything lives under fixed paths (/content/mamba, /content/TRELLIS.2, /content/Hunyuan3D-2.1) so a packed env
# works unchanged on another runtime. CUDA extensions are compiled for the A100 only (sm_80): that makes the
# build faster, and it means the build can run on a cheap T4 runtime even though the result only runs on an A100.
# NOT TESTED end to end before the first Colab run: expect to read the last lines of the log and adjust.
set -euo pipefail
export MAMBA_ROOT_PREFIX=/content/mamba
export TORCH_CUDA_ARCH_LIST="8.0"
export MAX_JOBS="${MAX_JOBS:-4}"
MM=/content/bin/micromamba
LOG=/content/install_logs
mkdir -p "$LOG"

step="${1:?usage: install_envs.sh micromamba|trellis2|hunyuan|pack|unpack [dir]}"

case "$step" in
  micromamba)
    mkdir -p /content/bin
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /content bin/micromamba
    $MM --version
    ;;

  trellis2)
    $MM create -y -n trellis2 -c conda-forge python=3.10 git ninja cmake 2>&1 | tail -3
    # CUDA 12.4 toolkit inside the env so nvcc matches torch 2.6.0+cu124 whatever Colab ships
    $MM install -y -n trellis2 -c nvidia/label/cuda-12.4.1 cuda-toolkit 2>&1 | tail -3
    export CUDA_HOME=/content/mamba/envs/trellis2
    export PATH="$CUDA_HOME/bin:$PATH"
    $MM run -n trellis2 pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -2
    rm -rf /content/TRELLIS.2
    git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive /content/TRELLIS.2
    cd /content/TRELLIS.2
    # same flags as the README, minus --new-env (the env already exists with the right torch)
    $MM run -n trellis2 bash -c "export CUDA_HOME=$CUDA_HOME PATH=$CUDA_HOME/bin:\$PATH; . ./setup.sh --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm" 2>&1 | tee "$LOG/trellis2_setup.log" | tail -25
    $MM run -n trellis2 pip install trimesh 2>&1 | tail -1
    $MM run -n trellis2 python -c "import torch, o_voxel, trellis2; print('trellis2 env ok, torch', torch.__version__)"
    ;;

  hunyuan)
    $MM create -y -n hunyuan -c conda-forge python=3.10 git ninja cmake 2>&1 | tail -3
    $MM install -y -n hunyuan -c nvidia/label/cuda-12.4.1 cuda-toolkit 2>&1 | tail -3
    export CUDA_HOME=/content/mamba/envs/hunyuan
    export PATH="$CUDA_HOME/bin:$PATH"
    $MM run -n hunyuan pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -2
    rm -rf /content/Hunyuan3D-2.1
    git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git /content/Hunyuan3D-2.1
    cd /content/Hunyuan3D-2.1
    $MM run -n hunyuan pip install -r requirements.txt 2>&1 | tee "$LOG/hunyuan_requirements.log" | tail -5
    (cd hy3dpaint/custom_rasterizer && $MM run -n hunyuan bash -c "export CUDA_HOME=$CUDA_HOME PATH=$CUDA_HOME/bin:\$PATH; pip install -e ." 2>&1 | tail -5)
    (cd hy3dpaint/DifferentiableRenderer && $MM run -n hunyuan bash compile_mesh_painter.sh 2>&1 | tail -5)
    mkdir -p hy3dpaint/ckpt
    wget -q -nc https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P hy3dpaint/ckpt
    $MM run -n hunyuan python -c "import torch; print('hunyuan env ok, torch', torch.__version__)"
    ls LICENSE* 2>/dev/null && head -40 LICENSE* || true
    ;;

  pack)
    dest="${2:?destination folder}"
    mkdir -p "$dest"
    apt-get -qq install -y zstd > /dev/null 2>&1 || true
    tar -C / -cf - content/mamba content/bin content/TRELLIS.2 content/Hunyuan3D-2.1 | zstd -T0 -3 > "$dest/w5_envs.tar.zst"
    ls -lh "$dest/w5_envs.tar.zst"
    ;;

  unpack)
    src="${2:?folder holding w5_envs.tar.zst}"
    apt-get -qq install -y zstd > /dev/null 2>&1 || true
    zstd -dc "$src/w5_envs.tar.zst" | tar -C / -xf -
    $MM --version
    ;;

  *) echo "unknown step $step"; exit 2 ;;
esac
