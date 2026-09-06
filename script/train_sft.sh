#!/bin/bash
#SBATCH --account=IscrC_HydroFM
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=videoxum_sft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --chdir=/leonardo/home/userexternal/aintini0/GRPOVideo
#SBATCH --output=/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/logs/sft-%j.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/logs/sft-%j.err

set -Eeuo pipefail
trap 'rc=$?; echo "[ERROR] train_sft.sh: riga $LINENO, exit=$rc, comando: $BASH_COMMAND" >&2' ERR

PROJECT_DIR="/leonardo/home/userexternal/aintini0/GRPOVideo"
DATA_ROOT="${DATA_ROOT:-/leonardo_work/IscrC_HydroFM/aintini0/datasets/videoxum}"
VENV_DIR="/leonardo_scratch/fast/IscrC_MSMU/aintini0/venvs/grpovideo"
MODEL_DIR="${MODEL_DIR:-/leonardo_work/IscrC_HydroFM/aintini0/models/Qwen3-VL-8B-Instruct}"
DATA_JSON="${DATA_JSON:-$DATA_ROOT/metadata/derived/train_videoxum_dpo.json}"
RAW_DATA_JSON="${RAW_DATA_JSON:-$DATA_ROOT/metadata/train_videoxum.json}"
VAL_JSON="${VAL_JSON:-$DATA_ROOT/metadata/val_videoxum.json}"
CHECKPOINT_ROOT="${SFT_CHECKPOINT_ROOT:-/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/checkpoints-sft}"
RESULTS_ROOT="${SFT_RESULTS_ROOT:-/leonardo_work/IscrC_HydroFM/aintini0/GRPOVideo/results/sft}"

cd "$PROJECT_DIR"

[[ -x "$VENV_DIR/bin/python" ]] || { echo "Virtualenv assente: $VENV_DIR" >&2; exit 2; }
[[ -d "$MODEL_DIR" ]] || { echo "Modello assente: $MODEL_DIR" >&2; exit 2; }
[[ -f "$DATA_JSON" ]] || { echo "Dataset DPO assente: $DATA_JSON" >&2; exit 2; }
[[ -f "$VAL_JSON" ]] || { echo "Validation assente: $VAL_JSON" >&2; exit 2; }
[[ -d "$DATA_ROOT/train" ]] || { echo "Video train assenti: $DATA_ROOT/train" >&2; exit 2; }
[[ -d "$DATA_ROOT/val" ]] || { echo "Video validation assenti: $DATA_ROOT/val" >&2; exit 2; }

module load python/3.11.7
source "/leonardo_scratch/fast/IscrC_MSMU/aintini0/venvs/grpovideo/bin/activate"

export HF_HOME="${SCRATCH}/GRPOVideo/cache/huggingface"
export TORCH_HOME="${SCRATCH}/GRPOVideo/cache/torch"
export PIP_CACHE_DIR="${SCRATCH}/GRPOVideo/cache/pip"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export MKL_THREADING_LAYER=GNU

mkdir -p "$CHECKPOINT_ROOT" "$RESULTS_ROOT" "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR"

EXTRA_ARGS=()
if [[ -n "${SFT_RESUME:-}" ]]; then
    [[ -d "$SFT_RESUME" ]] || { echo "Checkpoint assente: $SFT_RESUME" >&2; exit 2; }
    EXTRA_ARGS+=(--resume_from "$SFT_RESUME")
fi

echo "[INFO] model=$MODEL_DIR"
echo "[INFO] train_json=$DATA_JSON"
echo "[INFO] checkpoint=$CHECKPOINT_ROOT"
echo "[INFO] results=$RESULTS_ROOT"
echo "[INFO] python=$(command -v python)"

python --version

srun python -u sft_train_videoxum.py \
    --model "$MODEL_DIR" \
    --data_json "$DATA_JSON" \
    --video_dir "$DATA_ROOT/train" \
    --val_json "$VAL_JSON" \
    --val_video_dir "$DATA_ROOT/val" \
    --saveroot "$CHECKPOINT_ROOT" \
    --resultsroot "$RESULTS_ROOT" \
    --epochs "${SFT_EPOCHS:-1}" \
    --accum_steps "${SFT_ACCUM_STEPS:-4}" \
    --lr "${SFT_LR:-2e-4}" \
    --debug_samples "${SFT_TRAIN_SAMPLES:-4000}" \
    --eval_samples "${SFT_EVAL_SAMPLES:-1000}" \
    --attn_impl flash_attention_2 \
    --max_frames 0 \
    --save_steps "${SAVE_STEPS:-100}" \
    --device cuda:0 \
    "${EXTRA_ARGS[@]}"

echo "[INFO] SFT completata: $(date)"