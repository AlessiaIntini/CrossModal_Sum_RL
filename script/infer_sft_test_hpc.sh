#!/bin/bash
#SBATCH --job-name=videoxum_infer
#SBATCH --partition=gpu_a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --chdir=/home/aintini/RL_VideoSum/GRPOVideo
#SBATCH --output=/home/aintini/RL_VideoSum/GRPOVideo/logs/infer-%j.out
#SBATCH --error=/home/aintini/RL_VideoSum/GRPOVideo/logs/infer-%j.err

set -Eeuo pipefail
trap 'rc=$?; echo "[ERROR] infer_sft_test_hpc.sh: riga $LINENO, exit=$rc, comando: $BASH_COMMAND" >&2' ERR

PROJECT_DIR="/home/aintini/RL_VideoSum/GRPOVideo"
DATA_ROOT="${DATA_ROOT:-$PROJECT_DIR/dataset/aintini/datasets/videoxum}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/models/Qwen2.5-VL-7B-Instruct}"
CHECKPOINT="${CHECKPOINT:-${SFT_CHECKPOINT:-$PROJECT_DIR/outputs/sft/checkpoints/Qwen2.5-VL-7B-Instruct/checkpoint-best}}"
TEST_JSON="${TEST_JSON:-$DATA_ROOT/metadata/test_videoxum.json}"
TEST_VIDEO_DIR="${TEST_VIDEO_DIR:-$DATA_ROOT/test}"
OUTPUT_DIR="${INFERENCE_OUTPUT_DIR:-$PROJECT_DIR/outputs/inference}"

cd "$PROJECT_DIR"

[[ -x "$VENV_DIR/bin/python" ]] || { echo "Virtualenv assente: $VENV_DIR" >&2; exit 2; }
[[ -d "$MODEL_DIR" ]] || { echo "Modello base assente: $MODEL_DIR" >&2; exit 2; }
[[ -f "$CHECKPOINT/adapter_config.json" ]] || { echo "Checkpoint PEFT non valido: $CHECKPOINT" >&2; exit 2; }
[[ -f "$TEST_JSON" ]] || { echo "Metadata test assenti: $TEST_JSON" >&2; exit 2; }
[[ -d "$TEST_VIDEO_DIR" ]] || { echo "Video test assenti: $TEST_VIDEO_DIR" >&2; exit 2; }

source "$VENV_DIR/bin/activate"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_DIR/.cache/torch}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export MKL_THREADING_LAYER=GNU
mkdir -p "$OUTPUT_DIR" "$HF_HOME" "$TORCH_HOME"

echo "==================================================================="
echo "[INFO] Inferenza adapter PEFT sul test: $(date)"
echo "[INFO] job=${SLURM_JOB_ID:-local} host=$(hostname)"
echo "[INFO] checkpoint=$CHECKPOINT"
echo "[INFO] subset start=${TEST_START_INDEX:-0} samples=${TEST_SAMPLES:-0} (0=tutti)"
echo "[INFO] attention=${ATTN_IMPL:-sdpa}"
echo "[INFO] output=$OUTPUT_DIR"
python --version
nvidia-smi
srun python -c 'import sys, torch; print(f"[INFO] torch={torch.__version__} cuda_build={torch.version.cuda} cuda_available={torch.cuda.is_available()}"); sys.exit(0 if torch.cuda.is_available() else "PyTorch non riesce a inizializzare CUDA")'

srun python -u infer_sft_videoxum.py \
    --model "$MODEL_DIR" \
    --checkpoint "$CHECKPOINT" \
    --test_json "$TEST_JSON" \
    --test_video_dir "$TEST_VIDEO_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "${RUN_NAME:-}" \
    --start_index "${TEST_START_INDEX:-0}" \
    --num_samples "${TEST_SAMPLES:-0}" \
    --attn_impl "${ATTN_IMPL:-sdpa}" \
    --max_pixels "${MAX_PIXELS:-100352}" \
    --min_pixels "${MIN_PIXELS:-3136}" \
    --eval_max_new_tokens "${MAX_NEW_TOKENS:-256}" \
    --device cuda:0

echo "[INFO] Inferenza test completata: $(date)"
echo "==================================================================="
