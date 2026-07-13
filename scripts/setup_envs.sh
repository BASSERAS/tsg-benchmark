#!/bin/bash
# TSG Benchmark — Environment Setup
# Run: bash scripts/setup_envs.sh
set -e

# Install Miniconda if not available
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Installing Miniconda..."
    curl -sL "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -o /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p ./miniconda3
    eval "$(./miniconda3/bin/conda shell.bash hook)"
fi

echo "Creating conda environments..."

# TF1 env (TimeGAN, RGAN — TF1 compat mode)
conda create -y -n tf1_env python=3.7 pip
conda run -n tf1_env pip install tensorflow==2.2.0 numpy scikit-learn pandas tqdm matplotlib scipy

# Common PyTorch env (CSDI, TSDiff, Diffusion-TS, GT-GAN, Fourier-flows)
conda create -y -n common_pt python=3.10 pip
conda run -n common_pt pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
conda run -n common_pt pip install numpy pandas scikit-learn scipy matplotlib seaborn tqdm pyyaml einops opt_einsum
conda run -n common_pt pip install pytorch-lightning==1.9.5 gluonts torchdiffeq controldiffeq linear_attention_transformer pot

# timeVAE env (TF2)
conda create -y -n timevae_env python=3.12 pip
conda run -n timevae_env pip install tensorflow==2.16.1 numpy pandas scikit-learn matplotlib pyyaml

echo "All environments created!"
echo ""
echo "Clone method repos:"
for repo in \
    "https://github.com/jsyoon0823/TimeGAN" \
    "https://github.com/ratschlab/RGAN" \
    "https://github.com/Jinsung-Jeon/GT-GAN" \
    "https://github.com/abudesai/timeVAE" \
    "https://github.com/ahmedmalaa/Fourier-flows" \
    "https://github.com/ermongroup/CSDI" \
    "https://github.com/amazon-science/unconditional-time-series-diffusion" \
    "https://github.com/Y-debug-sys/Diffusion-TS"; do
    name=$(basename "$repo")
    if [ ! -d "repos/$name" ]; then
        echo "  git clone $repo repos/$name"
    fi
done
