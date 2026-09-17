#!/bin/bash
#SBATCH --account=IscrC_HydroFM
#SBATCH --partition=lrd_all_serial
#SBATCH --job-name=download-clip
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --chdir=/leonardo/home/userexternal/aintini0/GRPOVideo
#SBATCH --output=/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/logs/download-clip-%j.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/logs/download-clip-%j.err

set -Eeuo pipefail
MODEL_ID="openai/clip-vit-large-patch14-336"
MODEL_DIR="${MODEL_DIR:-/leonardo_work/IscrC_HydroFM/aintini0/models/clip-vit-large-patch14-336}"
VENV_DIR="${VENV_DIR:-/leonardo_scratch/fast/IscrC_MSMU/aintini0/venvs/grpovideo}"

module load python/3.11.7
source "$VENV_DIR/bin/activate"
mkdir -p "$MODEL_DIR"
hf download "$MODEL_ID" --local-dir "$MODEL_DIR"
test -f "$MODEL_DIR/config.json"
echo "[DONE] Modello CLIP salvato in $MODEL_DIR"
