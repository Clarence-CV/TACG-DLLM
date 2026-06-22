#!/bin/bash
set -euo pipefail

echo "[1/4] Upgrading pip tooling"
python -m pip install --upgrade pip setuptools wheel

echo "[2/4] Installing Python dependencies"
pip install -r requirements.txt

echo "[3/4] Ensuring Hugging Face CLI is available"
pip install "huggingface_hub[cli]"

echo "[4/4] Optional model download locations"
echo "Expected local model paths used by scripts by default:"
echo "  - /data/ckpt/LLaDa-8B"
echo "  - /data/ckpt/Dream-v0-Instruct-7B"
echo
echo "If you want to download models manually, for example:"
echo "  huggingface-cli download GSAI-ML/LLaDA-8B-Instruct --local-dir /data/ckpt/LLaDa-8B"
echo "  huggingface-cli download Dream-org/Dream-v0-Instruct-7B --local-dir /data/ckpt/Dream-v0-Instruct-7B"
echo
echo "Installation finished."
echo "You can now run scripts from ./scripts/."
