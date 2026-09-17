#!/bin/bash
#SBATCH --account=IscrC_HydroFM
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=eval-test-vsum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --chdir=/leonardo/home/userexternal/aintini0/GRPOVideo
#SBATCH --output=/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/logs/eval-test-vsum-%j.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/logs/eval-test-vsum-%j.err

set -Eeuo pipefail

VENV_DIR="${VENV_DIR:-/leonardo_scratch/fast/IscrC_MSMU/aintini0/venvs/grpovideo}"
PREDICTIONS="${PREDICTIONS:-/leonardo_work/IscrC_MSMU/aintini0/V2Xum-LLM/results/videoxum_test_vsum}"
ANNOTATIONS="${ANNOTATIONS:-/leonardo_work/IscrC_HydroFM/aintini0/datasets/videoxum/metadata/test_videoxum.json}"
OUTPUT="${OUTPUT:-/leonardo_work/IscrC_HydroFM/aintini0/GRPOVideo/results/evaluation/videoxum_test_vsum_metrics.jsonl}"

module load python/3.11.7
source "$VENV_DIR/bin/activate"
mkdir -p "$(dirname "$OUTPUT")"

srun python -u utils.py \
    --predictions "$PREDICTIONS" \
    --annotations "$ANNOTATIONS" \
    --output "$OUTPUT" \
    --index-base 1
