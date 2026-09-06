#!/bin/bash
#SBATCH --account=IscrC_HydroFM
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=videoxum_dpo_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --chdir=/leonardo/home/userexternal/aintini0/GRPOVideo
#SBATCH --output=/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/logs/infer-dpo-%j.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/logs/infer-dpo-%j.err

set -Eeuo pipefail
trap 'rc=$?; echo "[ERROR] infer_dpo_test_cineca.sh: riga $LINENO, exit=$rc, comando: $BASH_COMMAND" >&2' ERR

PROJECT_DIR="/leonardo/home/userexternal/aintini0/GRPOVideo"
VENV_DIR="${VENV_DIR:-/leonardo_scratch/fast/IscrC_MSMU/aintini0/venvs/grpovideo}"
DATA_ROOT="${DATA_ROOT:-/leonardo_work/IscrC_HydroFM/aintini0/datasets/videoxum}"
MODEL_DIR="${MODEL_DIR:-/leonardo_work/IscrC_HydroFM/aintini0/models/Qwen3-VL-8B-Instruct}"
CHECKPOINT="${CHECKPOINT:-${DPO_CHECKPOINT:-/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/checkpoints/Qwen3-VL-8B-Instruct/checkpoint-best}}"
TEST_JSON="${TEST_JSON:-$DATA_ROOT/metadata/test_videoxum.json}"
TEST_VIDEO_DIR="${TEST_VIDEO_DIR:-$DATA_ROOT/test}"
OUTPUT_DIR="${INFERENCE_OUTPUT_DIR:-/leonardo_work/IscrC_HydroFM/aintini0/GRPOVideo/results/inference-dpo}"
SCRATCH_ROOT="${SCRATCH:-/leonardo_scratch/fast/IscrC_HydroFM/aintini0}"

cd "$PROJECT_DIR"

[[ -x "$VENV_DIR/bin/python" ]] || { echo "[ERROR] Virtualenv assente: $VENV_DIR" >&2; exit 2; }
[[ -d "$MODEL_DIR" ]] || { echo "[ERROR] Modello base assente: $MODEL_DIR" >&2; exit 2; }
[[ -f "$MODEL_DIR/config.json" ]] || { echo "[ERROR] config.json del modello assente: $MODEL_DIR" >&2; exit 2; }
[[ -f "$CHECKPOINT/adapter_config.json" ]] || { echo "[ERROR] Checkpoint DPO non valido: $CHECKPOINT" >&2; exit 2; }
[[ -f "$TEST_JSON" ]] || { echo "[ERROR] Metadata test assenti: $TEST_JSON" >&2; exit 2; }
[[ -d "$TEST_VIDEO_DIR" ]] || { echo "[ERROR] Video test assenti: $TEST_VIDEO_DIR" >&2; exit 2; }

module load python/3.11.7
source "$VENV_DIR/bin/activate"

export HF_HOME="${HF_HOME:-$SCRATCH_ROOT/GRPOVideo/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$SCRATCH_ROOT/GRPOVideo/cache/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SCRATCH_ROOT/GRPOVideo/cache/pip}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export MKL_THREADING_LAYER=GNU
mkdir -p "$OUTPUT_DIR" "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR"

echo "==================================================================="
echo "[INFO] Inferenza DPO sul test: $(date)"
echo "[INFO] job=${SLURM_JOB_ID:-local} host=$(hostname)"
echo "[INFO] model=$MODEL_DIR"
echo "[INFO] checkpoint=$CHECKPOINT"
echo "[INFO] test_json=$TEST_JSON"
echo "[INFO] test_video_dir=$TEST_VIDEO_DIR"
echo "[INFO] output=$OUTPUT_DIR"
echo "[INFO] subset start=${TEST_START_INDEX:-0} samples=${TEST_SAMPLES:-0} (0=tutti)"
echo "[INFO] attention=${ATTN_IMPL:-flash_attention_2}"
echo "[INFO] python=$(command -v python)"
python --version
nvidia-smi

srun python -c 'import sys, torch; print(f"[INFO] torch={torch.__version__} cuda_build={torch.version.cuda} cuda_available={torch.cuda.is_available()}"); sys.exit(0 if torch.cuda.is_available() else "PyTorch non riesce a inizializzare CUDA")'

srun python -u infer_sft_videoxum.py \
    --model "$MODEL_DIR" \
    --checkpoint "$CHECKPOINT" \
    --test_json "$TEST_JSON" \
    --test_video_dir "$TEST_VIDEO_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "${RUN_NAME:-dpo-test}" \
    --start_index "${TEST_START_INDEX:-0}" \
    --num_samples "${TEST_SAMPLES:-0}" \
    --attn_impl "${ATTN_IMPL:-flash_attention_2}" \
    --eval_max_new_tokens "${MAX_NEW_TOKENS:-256}" \
    --device cuda:0

echo "==================================================================="
echo "[INFO] Inferenza DPO completata: $(date)"
echo "==================================================================="
