#!/bin/bash
#SBATCH --job-name=videoxum_dpo
#SBATCH --partition=gpu_a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/aintini/RL_VideoSum/GRPOVideo
#SBATCH --output=/home/aintini/RL_VideoSum/GRPOVideo/logs/dpo-%j.out
#SBATCH --error=/home/aintini/RL_VideoSum/GRPOVideo/logs/dpo-%j.err

set -Eeuo pipefail
trap 'rc=$?; echo "[ERROR] train_dpo_hpc.sh: riga $LINENO, exit=$rc, comando: $BASH_COMMAND" >&2' ERR

PROJECT_DIR="/home/aintini/RL_VideoSum/GRPOVideo"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/dataset/aintini/datasets/videoxum}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models/Qwen2.5-VL-7B-Instruct}"
DATA_JSON="${DATA_JSON:-$DATA_ROOT/metadata/derived/train_videoxum_dpo.json}"
RAW_DATA_JSON="${RAW_DATA_JSON:-$DATA_ROOT/metadata/train_videoxum.json}"
VAL_JSON="${VAL_JSON:-$DATA_ROOT/metadata/val_videoxum.json}"
CHECKPOINT_ROOT="${DPO_CHECKPOINT_ROOT:-$PROJECT_DIR/outputs/dpo/checkpoints}"
RESULTS_ROOT="${DPO_RESULTS_ROOT:-$PROJECT_DIR/outputs/dpo/results}"

cd "$PROJECT_DIR"
echo "==================================================================="
echo "[INFO] Avvio DPO: $(date)"
echo "[INFO] job=${SLURM_JOB_ID:-local} host=$(hostname)"
echo "[INFO] partition=${SLURM_JOB_PARTITION:-N/A} gpu=${CUDA_VISIBLE_DEVICES:-N/A}"

[[ -x "$VENV_DIR/bin/python" ]] || { echo "Virtualenv assente: $VENV_DIR" >&2; exit 2; }
[[ -d "$MODEL_DIR" ]] || { echo "Modello base assente: $MODEL_DIR (impostare MODEL_DIR)" >&2; exit 2; }
[[ -f "$RAW_DATA_JSON" ]] || { echo "Metadata train assenti: $RAW_DATA_JSON" >&2; exit 2; }
[[ -f "$VAL_JSON" ]] || { echo "Metadata validation assenti: $VAL_JSON" >&2; exit 2; }
[[ -d "$DATA_ROOT/train" ]] || { echo "Video train assenti: $DATA_ROOT/train" >&2; exit 2; }
[[ -d "$DATA_ROOT/val" ]] || { echo "Video validation assenti: $DATA_ROOT/val" >&2; exit 2; }

source "$VENV_DIR/bin/activate"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_DIR/.cache/torch}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export MKL_THREADING_LAYER=GNU
mkdir -p "$CHECKPOINT_ROOT" "$RESULTS_ROOT" "$HF_HOME" "$TORCH_HOME"

if [[ ! -f "$DATA_JSON" ]]; then
    mkdir -p "$(dirname "$DATA_JSON")"
    echo "[INFO] Creo le preferenze DPO in $DATA_JSON"
    python -u configuration/createDPODate.py \
        --input "$RAW_DATA_JSON" \
        --output "$DATA_JSON"
fi

EXTRA_ARGS=()
if [[ -n "${DPO_RESUME:-}" ]]; then
    [[ -d "$DPO_RESUME" ]] || { echo "Checkpoint DPO assente: $DPO_RESUME" >&2; exit 2; }
    EXTRA_ARGS+=(--resume_from "$DPO_RESUME")
fi

echo "[INFO] model=$MODEL_DIR"
echo "[INFO] train_json=$DATA_JSON"
echo "[INFO] checkpoint=$CHECKPOINT_ROOT"
echo "[INFO] results=$RESULTS_ROOT"
echo "[INFO] resume=${DPO_RESUME:-nessuno}"
echo "[INFO] train_samples=${DPO_TRAIN_SAMPLES:-4000}"
echo "[INFO] eval_samples=${DPO_EVAL_SAMPLES:-1000}"
echo "[INFO] python=$(command -v python)"
python --version
nvidia-smi
srun python -c 'import sys, torch; print(f"[INFO] torch={torch.__version__} cuda_build={torch.version.cuda} cuda_available={torch.cuda.is_available()}"); sys.exit(0 if torch.cuda.is_available() else "PyTorch non riesce a inizializzare CUDA")'

srun python -u train_dpo.py \
    --model "$MODEL_DIR" --data_json "$DATA_JSON" \
    --video_dir "$DATA_ROOT/train" --val_json "$VAL_JSON" \
    --val_video_dir "$DATA_ROOT/val" --saveroot "$CHECKPOINT_ROOT" \
    --resultsroot "$RESULTS_ROOT" --epochs "${DPO_EPOCHS:-1}" \
    --accum_steps "${DPO_ACCUM_STEPS:-16}" --lr "${DPO_LR:-5e-6}" \
    --beta "${DPO_BETA:-0.1}" \
    --debug_samples "${DPO_TRAIN_SAMPLES:-4000}" \
    --eval_samples "${DPO_EVAL_SAMPLES:-1000}" \
    --max_pixels "${MAX_PIXELS:-100352}" \
    --min_pixels "${MIN_PIXELS:-3136}" --max_frames 0 \
    --save_steps "${SAVE_STEPS:-100}" --device cuda:0 \
    "${EXTRA_ARGS[@]}"

echo "==================================================================="
echo "[INFO] DPO completato: $(date)"
echo "==================================================================="
