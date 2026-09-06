#!/bin/bash
#SBATCH --account=IscrC_HydroFM
#SBATCH --partition=lrd_all_serial
#SBATCH --job-name=create_dpo_data
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/logs/create-dpo-data-%j.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/logs/create-dpo-data-%j.err

set -euo pipefail

PROJECT_DIR="/leonardo/home/userexternal/aintini0/GRPOVideo"
VENV_DIR="/leonardo_scratch/fast/IscrC_HydroFM/aintini0/venvs/grpovideo"

cd "$PROJECT_DIR"
module load python/3.11.7
source "$VENV_DIR/bin/activate"

export HF_HOME="${SCRATCH}/GRPOVideo/cache/huggingface"
export PIP_CACHE_DIR="${SCRATCH}/GRPOVideo/cache/pip"

python createDPODate.py "$@"
