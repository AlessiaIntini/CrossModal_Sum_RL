#!/bin/bash
#SBATCH --account=IscrC_HydroFM
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=videoxum_grpo_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --chdir=/leonardo/home/userexternal/aintini0/GRPOVideo
#SBATCH --output=/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/logs/infer-grpo-dpo-ConsensusDataset-onlySummary-%j.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/logs/infer-grpo-dpo-ConsensusDataset-onlySummary-%j.err
set -Eeuo pipefail
trap 'rc=$?; echo "[ERROR] infer_grpo_test_cineca.sh: riga $LINENO, exit=$rc, comando: $BASH_COMMAND" >&2' ERR
PROJECT_DIR="/leonardo/home/userexternal/aintini0/GRPOVideo"
VENV_DIR="${VENV_DIR:-/leonardo_scratch/fast/IscrC_MSMU/aintini0/venvs/grpovideo}"
DATA_ROOT="${DATA_ROOT:-/leonardo_work/IscrC_HydroFM/aintini0/datasets/videoxum}"
MODEL_DIR="${MODEL_DIR:-/leonardo_work/IscrC_HydroFM/aintini0/models/Qwen3-VL-8B-Instruct}"
CHECKPOINT="${CHECKPOINT:-${GRPO_CHECKPOINT:-/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/checkpoints-grpo-dpo-consensus-v2/Qwen3-VL-8B-Instruct/checkpoint-epoch1-preval}}"
TEST_JSON="${TEST_JSON:-$DATA_ROOT/metadata/test_videoxum.json}"
TEST_VIDEO_DIR="${TEST_VIDEO_DIR:-$DATA_ROOT/test}"
OUTPUT_DIR="${INFERENCE_OUTPUT_DIR:-/leonardo_work/IscrC_HydroFM/aintini0/GRPOVideo/results/inference-grpo-dpo-ConsensusDataset-onlySummary}"
SCRATCH_ROOT="${SCRATCH:-/leonardo_scratch/fast/IscrC_HydroFM/aintini0}"
EXTRA_ARGS=()
# SUMMARY_STYLE deve seguire il checkpoint: concise (default) per i checkpoint GRPO
# addestrati prima del 2026-09-17, reference per quelli nuovi.
if [[ "${SUMMARY_ONLY:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--summary_only)
fi
# Adapter SFT/DPO da fondere prima del LoRA GRPO. I checkpoint GRPO nuovi lo
# registrano in grpo_init.json; per quelli precedenti va passato INIT_FROM,
# altrimenti il LoRA GRPO viene valutato sul Qwen base.
[[ -n "${INIT_FROM:-}" ]] && EXTRA_ARGS+=(--init_from "$INIT_FROM")
# max_pixels/min_pixels/timeline_unit: se non impostati si ereditano dal
# checkpoint; i checkpoint GRPO precedenti erano addestrati con MAX_PIXELS=100352.
[[ -n "${MAX_PIXELS:-}" ]] && EXTRA_ARGS+=(--max_pixels "$MAX_PIXELS")
[[ -n "${MIN_PIXELS:-}" ]] && EXTRA_ARGS+=(--min_pixels "$MIN_PIXELS")
[[ -n "${TIMELINE_UNIT:-}" ]] && EXTRA_ARGS+=(--timeline_unit "$TIMELINE_UNIT")
cd "$PROJECT_DIR"
[[ -x "$VENV_DIR/bin/python" ]] || { echo "[ERROR] Virtualenv assente: $VENV_DIR" >&2; exit 2; }
[[ -f "$MODEL_DIR/config.json" ]] || { echo "[ERROR] Modello base non valido: $MODEL_DIR" >&2; exit 2; }
[[ -f "$CHECKPOINT/adapter_config.json" ]] || { echo "[ERROR] Checkpoint GRPO finale non valido: $CHECKPOINT" >&2; exit 2; }
[[ -f "$TEST_JSON" ]] || { echo "[ERROR] Metadata test assenti: $TEST_JSON" >&2; exit 2; }
[[ -d "$TEST_VIDEO_DIR" ]] || { echo "[ERROR] Video test assenti: $TEST_VIDEO_DIR" >&2; exit 2; }
module load python/3.11.7
source "$VENV_DIR/bin/activate"
export HF_HOME="${HF_HOME:-$SCRATCH_ROOT/GRPOVideo/cache/huggingface}" TORCH_HOME="${TORCH_HOME:-$SCRATCH_ROOT/GRPOVideo/cache/torch}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 MKL_THREADING_LAYER=GNU
mkdir -p "$OUTPUT_DIR" "$HF_HOME" "$TORCH_HOME"
echo "[INFO] GRPO checkpoint=$CHECKPOINT"
nvidia-smi
srun python -u infer_grpo_videoxum.py \
    --model "$MODEL_DIR" --checkpoint "$CHECKPOINT" \
    --test_json "$TEST_JSON" --test_video_dir "$TEST_VIDEO_DIR" \
    --output_dir "$OUTPUT_DIR" --run_name "${RUN_NAME:-grpo-final-test}" \
    --start_index "${TEST_START_INDEX:-0}" --num_samples "${TEST_SAMPLES:-0}" \
    --attn_impl "${ATTN_IMPL:-flash_attention_2}" \
    --max_completion_length "${MAX_NEW_TOKENS:-384}" \
    --summary_style "${SUMMARY_STYLE:-concise}" \
    --device cuda:0 "${EXTRA_ARGS[@]}"
echo "[INFO] Inferenza GRPO completata: $(date)"
