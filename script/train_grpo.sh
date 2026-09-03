#!/bin/bash
#SBATCH --job-name=videoxum_grpo
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=alessia.intini@polito.it
#SBATCH --partition=gpu_a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/aintini/RL_VideoSum/GRPOVideo
#SBATCH --output=/home/aintini/RL_VideoSum/GRPOVideo/logs/grpo-%j.out
#SBATCH --error=/home/aintini/RL_VideoSum/GRPOVideo/logs/grpo-%j.err

set -Eeuo pipefail
trap 'rc=$?; echo "[ERROR] train_grpo.sh: riga $LINENO, exit=$rc, comando: $BASH_COMMAND" >&2' ERR

PROJECT_DIR="/home/aintini/RL_VideoSum/GRPOVideo"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/dataset/aintini/datasets/videoxum}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models/Qwen2.5-VL-3B-Instruct}"
DATA_JSON="${DATA_JSON:-$DATA_ROOT/metadata/train_videoxum.json}"
VAL_JSON="${VAL_JSON:-$DATA_ROOT/metadata/val_videoxum.json}"
FEATURE_DIR="${FEATURE_DIR:-$DATA_ROOT/feat/blip}"
CHECKPOINT_ROOT="${GRPO_CHECKPOINT_ROOT:-$PROJECT_DIR/outputs/grpo/checkpoints}"
RESULTS_ROOT="${GRPO_RESULTS_ROOT:-$PROJECT_DIR/outputs/grpo/results}"

cd "$PROJECT_DIR"
echo "==================================================================="
echo "[INFO] Avvio GRPO: $(date)"
echo "[INFO] job=${SLURM_JOB_ID:-local} host=$(hostname)"
echo "[INFO] partition=${SLURM_JOB_PARTITION:-N/A} gpu=${CUDA_VISIBLE_DEVICES:-N/A}"

[[ -x "$VENV_DIR/bin/python" ]] || { echo "Virtualenv assente: $VENV_DIR" >&2; exit 2; }
[[ -d "$MODEL_DIR" ]] || { echo "Modello base assente: $MODEL_DIR (impostare MODEL_DIR)" >&2; exit 2; }
[[ -f "$DATA_JSON" ]] || { echo "Metadata GRPO assente: $DATA_JSON; rieseguire il notebook dataset_dataset.ipynb" >&2; exit 2; }
[[ -f "$VAL_JSON" ]] || { echo "Metadata validation assente: $VAL_JSON" >&2; exit 2; }
[[ -d "$FEATURE_DIR" ]] || { echo "Feature visuali assenti: $FEATURE_DIR (impostare FEATURE_DIR)" >&2; exit 2; }

source "$VENV_DIR/bin/activate"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_DIR/.cache/torch}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export MKL_THREADING_LAYER=GNU
mkdir -p "$CHECKPOINT_ROOT" "$RESULTS_ROOT" "$HF_HOME" "$TORCH_HOME"

# Tre modalita':
#   senza DPO: sbatch train_grpo.sh
#   con DPO:   sbatch --export=ALL,DPO_CHECKPOINT=/path/to/dpo/final train_grpo.sh
#   resume:    aggiungere GRPO_RESUME=/path/to/grpo/checkpoint
#   subset:    aggiungere TRAIN_SAMPLES=1000,VAL_SAMPLES=200 (0 = usa tutto)
EXTRA_ARGS=()
if [[ -n "${DPO_CHECKPOINT:-}" ]]; then
    [[ -d "$DPO_CHECKPOINT" ]] || { echo "Checkpoint DPO assente: $DPO_CHECKPOINT" >&2; exit 2; }
    EXTRA_ARGS+=(--init_from "$DPO_CHECKPOINT")
fi
if [[ -n "${GRPO_RESUME:-}" ]]; then
    [[ -d "$GRPO_RESUME" ]] || { echo "Checkpoint GRPO assente: $GRPO_RESUME" >&2; exit 2; }
    EXTRA_ARGS+=(--resume_from "$GRPO_RESUME")
fi

echo "job=${SLURM_JOB_ID:-local} node=$(hostname)"
echo "model=$MODEL_DIR"
echo "data=$DATA_ROOT"
echo "features=$FEATURE_DIR"
echo "DPO=${DPO_CHECKPOINT:-nessuno; partenza dal modello base}"
echo "python=$(command -v python)"
python --version
nvidia-smi

srun python -u train_grpo.py \
    --model "$MODEL_DIR" \
    --data_json "$DATA_JSON" \
    --video_dir "$DATA_ROOT/train" \
    --feature_dir "$FEATURE_DIR" \
    --val_json "$VAL_JSON" \
    --val_video_dir "$DATA_ROOT/val" \
    --val_feature_dir "$FEATURE_DIR" \
    --saveroot "$CHECKPOINT_ROOT" \
    --resultsroot "$RESULTS_ROOT" \
    --semantic_model "${SEMANTIC_MODEL:-sentence-transformers/all-MiniLM-L6-v2}" \
    --semantic_device "${SEMANTIC_DEVICE:-cpu}" \
    --epochs "${EPOCHS:-2}" \
    --accum_steps "${ACCUM_STEPS:-8}" \
    --num_generations "${NUM_GENERATIONS:-4}" \
    --lr "${LR:-1e-6}" \
    --beta "${BETA:-0.04}" \
    --max_completion_length "${MAX_COMPLETION_LENGTH:-256}" \
    --max_pixels "${MAX_PIXELS:-100352}" \
    --train_samples "${TRAIN_SAMPLES:-0}" \
    --val_samples "${VAL_SAMPLES:-200}" \
    --save_steps "${SAVE_STEPS:-100}" \
    --device cuda:0 \
    "${EXTRA_ARGS[@]}"

echo "==================================================================="
echo "[INFO] GRPO completato: $(date)"
echo "==================================================================="
