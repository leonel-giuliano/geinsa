#!/usr/bin/env bash
set -e

# Ensure conda is available in this shell
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found in PATH. Run 'conda init bash' and restart your terminal."
    exit 1
fi

# Load conda functions (required for conda activate to work in scripts)
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create the environment if it doesn’t exist
if ! conda env list | grep -q '^geinsa'; then
    echo "🌱 Creating new environment: geinsa"
    conda create -n geinsa python=3.11 -y
else
    echo "⚙️  Environment 'geinsa' already exists — skipping creation"
fi

# Activate it
conda activate geinsa

# Install CPU-only PyTorch via pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Core conda dependencies
conda install joblib pandas -y

# Python dependencies via pip
pip install firebase-admin sentence-transformers

# LightFM via conda-forge
conda install -c conda-forge lightfm -y

echo ""
echo "Enviroment 'geinsa' ready"
echo "Activate it manually with:"
echo "    conda activate geinsa"
