#!/bin/bash
#SBATCH --account=IscrC_MSMU
#SBATCH --partition=lrd_all_serial
#SBATCH --job-name=download_qwen
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/leonardo_scratch/fast/IscrC_MSMU/aintini0/GRPOVideo/logs/download-qwen-%j.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_MSMU/aintini0/GRPOVideo/logs/download-qwen-%j.err

set -euo pipefail

cd /leonardo/home/userexternal/aintini0/GRPOVideo
module load python/3.11.7
source /leonardo_scratch/fast/IscrC_MSMU/aintini0/venvs/grpovideo/bin/activate

export HF_HOME="${SCRATCH}/GRPOVideo/cache/huggingface"
export PIP_CACHE_DIR="${SCRATCH}/GRPOVideo/cache/pip"

MODEL_DIR="$WORK/$USER/models/Qwen2.5-VL-3B-Instruct"

mkdir -p "$MODEL_DIR"

hf download Qwen/Qwen2.5-VL-3B-Instruct \
    --local-dir "$MODEL_DIR"

echo "Modello salvato in: $MODEL_DIR"
du -sh "$MODEL_DIR"
