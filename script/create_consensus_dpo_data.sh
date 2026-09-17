#!/bin/bash
#SBATCH --account=IscrC_HydroFM
#SBATCH --partition=lrd_all_serial
#SBATCH --job-name=create_consensus_dpo_data
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/logs/create-consensus-dpo-data-%j.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/logs/create-consensus-dpo-data-%j.err

set -euo pipefail

PROJECT_DIR="/leonardo/home/userexternal/aintini0/GRPOVideo"
VENV_DIR="/leonardo_scratch/fast/IscrC_MSMU/aintini0/venvs/grpovideo"

cd "$PROJECT_DIR"
module load python/3.11.7
source "$VENV_DIR/bin/activate"

export HF_HOME="${SCRATCH}/GRPOVideo/cache/huggingface"
export PIP_CACHE_DIR="${SCRATCH}/GRPOVideo/cache/pip"

# TIMELINE_UNIT=pair: label e coppie DPO su ceil(N/2) segmenti, come le posizioni
# temporali di Qwen3-VL (vsum_onehot resta su N frame per le metriche).
# TIMELINE_UNIT=frame riproduce il dataset consensus precedente.
DATA_ROOT="${DATA_ROOT:-/leonardo_work/IscrC_HydroFM/aintini0/datasets/videoxum}"
TIMELINE_UNIT="${TIMELINE_UNIT:-pair}"
RAW_DATA_JSON="${RAW_DATA_JSON:-$DATA_ROOT/metadata/train_videoxum.json}"
if [[ "$TIMELINE_UNIT" == "pair" ]]; then
    DEFAULT_OUTPUT="$DATA_ROOT/metadata/consensus/train_videoxum_dpo_consensus_pairs.json"
else
    DEFAULT_OUTPUT="$DATA_ROOT/metadata/consensus/train_videoxum_dpo_consensus.json"
fi
OUTPUT_JSON="${OUTPUT_JSON:-$DEFAULT_OUTPUT}"
mkdir -p "$(dirname "$OUTPUT_JSON")"
echo "[INFO] timeline_unit=$TIMELINE_UNIT input=$RAW_DATA_JSON output=$OUTPUT_JSON"

srun python -u configuration/createConsensusDPOData.py \
    --input "$RAW_DATA_JSON" --output "$OUTPUT_JSON" \
    --timeline-unit "$TIMELINE_UNIT" \
    --hard-ratio 0.45 --intermediate-ratio 0.25 --overcoverage-ratio 0.30 \
    --overcoverage-band-ratios 0.30 0.50 0.20

