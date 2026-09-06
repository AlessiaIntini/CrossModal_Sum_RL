#!/bin/bash
#SBATCH --account=IscrC_HydroFM
#SBATCH --partition=lrd_all_serial
#SBATCH --job-name=download_blip
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --chdir=/leonardo/home/userexternal/aintini0/GRPOVideo
#SBATCH --output=/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/logs/download-blip-%j.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/logs/download-blip-%j.err

set -Eeuo pipefail
trap 'rc=$?; echo "[ERROR] download_blip.sh: riga $LINENO, exit=$rc, comando: $BASH_COMMAND" >&2' ERR

PROJECT_DIR="/leonardo/home/userexternal/aintini0/GRPOVideo"
VENV_DIR="${VENV_DIR:-/leonardo_scratch/fast/IscrC_MSMU/aintini0/venvs/grpovideo}"
MODEL_DIR="${MODEL_DIR:-/leonardo_work/IscrC_HydroFM/aintini0/models/blip-image-captioning-base}"
MODEL_ID="Salesforce/blip-image-captioning-base"

cd "$PROJECT_DIR"
module load python/3.11.7
source "$VENV_DIR/bin/activate"

export HF_HOME="${HF_HOME:-/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/cache/huggingface}"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$MODEL_DIR" "$HF_HOME"

echo "[INFO] Download di $MODEL_ID"
echo "[INFO] Destinazione: $MODEL_DIR"

# Il job e' seriale, quindi puo' raggiungere Hugging Face; il job GPU usera'
# invece esclusivamente questa copia locale.
hf download "$MODEL_ID" --local-dir "$MODEL_DIR"

test -f "$MODEL_DIR/config.json"
echo "[INFO] Modello scaricato correttamente: $MODEL_DIR"
du -sh "$MODEL_DIR"
