#!/bin/bash
#SBATCH --account=IscrC_HydroFM
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=videoxum_sft-v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --chdir=/leonardo/home/userexternal/aintini0/GRPOVideo
#SBATCH --output=/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/logs/sft-ConsensusDataset-2-%j.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/logs/sft-ConsensusDataset-2-%j.err

set -Eeuo pipefail
trap 'rc=$?; echo "[ERROR] train_sft.sh: riga $LINENO, exit=$rc, comando: $BASH_COMMAND" >&2' ERR

PROJECT_DIR="/leonardo/home/userexternal/aintini0/GRPOVideo"
DATA_ROOT="${DATA_ROOT:-/leonardo_work/IscrC_HydroFM/aintini0/datasets/videoxum}"
VENV_DIR="/leonardo_scratch/fast/IscrC_MSMU/aintini0/venvs/grpovideo"
MODEL_DIR="${MODEL_DIR:-/leonardo_work/IscrC_HydroFM/aintini0/models/Qwen3-VL-8B-Instruct}"
DATA_JSON="${DATA_JSON:-/leonardo_work/IscrC_HydroFM/aintini0/datasets/videoxum/metadata/consensus/train_videoxum_dpo_consensus_pairs.json}"
RAW_DATA_JSON="${RAW_DATA_JSON:-$DATA_ROOT/metadata/train_videoxum.json}"
VAL_JSON="${VAL_JSON:-$DATA_ROOT/metadata/val_videoxum.json}"
VAL_VIDEO_DIR="${VAL_VIDEO_DIR:-$DATA_ROOT/val}"
CHECKPOINT_ROOT="${SFT_CHECKPOINT_ROOT:-/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/checkpoints-sft-consensus-pairs}"
RESULTS_ROOT="${SFT_RESULTS_ROOT:-/leonardo_work/IscrC_HydroFM/aintini0/GRPOVideo/results/sft-consensus-pairs}"
# Configurazione video condivisa da SFT, DPO, GRPO e inferenza: viene salvata in
# video_encoding.json e le fasi successive la ereditano (o falliscono se diversa).
TIMELINE_UNIT="${TIMELINE_UNIT:-pair}"
MAX_PIXELS="${MAX_PIXELS=100352}"
MIN_PIXELS="${MIN_PIXELS:-3136}"

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

TIME_BUDGET_HOURS="${TIME_BUDGET_HOURS:-21.5}"
EXTRA_ARGS=()
if [[ -n "${SFT_RESUME:-}" ]]; then
    [[ -d "$SFT_RESUME" ]] || { echo "Checkpoint assente: $SFT_RESUME" >&2; exit 2; }
    EXTRA_ARGS+=(--resume_from "$SFT_RESUME")
fi

echo "[INFO] model=$MODEL_DIR"
echo "[INFO] train_json=$DATA_JSON"
echo "[INFO] checkpoint=$CHECKPOINT_ROOT"
echo "[INFO] results=$RESULTS_ROOT"
echo "[INFO] resume=${SFT_RESUME:-nessuno}"
echo "[INFO] train_samples=${SFT_TRAIN_SAMPLES:-4000}"
echo "[INFO] eval_samples=${SFT_EVAL_SAMPLES:-300} eval_steps=${EVAL_STEPS:-250}"
echo "[INFO] video_time_mode=${VIDEO_TIME_MODE:-auto} time_budget_hours=$TIME_BUDGET_HOURS"
echo "[INFO] timeline_unit=$TIMELINE_UNIT max_pixels=$MAX_PIXELS min_pixels=$MIN_PIXELS best_metric=${BEST_METRIC:-default}"
if [[ -n "${BEST_METRIC:-}" ]]; then
    EXTRA_ARGS+=(--best_metric "$BEST_METRIC")
fi
echo "[INFO] python=$(command -v python)"

python --version

rc=0
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
    --eval_samples "${SFT_EVAL_SAMPLES:-300}" \
    --eval_steps "${EVAL_STEPS:-250}" \
    --eval_steps_samples "${EVAL_STEPS_SAMPLES:-200}" \
    --video_time_mode "${VIDEO_TIME_MODE:-auto}" \
    --timeline_unit "$TIMELINE_UNIT" \
    --max_pixels "$MAX_PIXELS" \
    --min_pixels "$MIN_PIXELS" \
    --time_budget_hours "$TIME_BUDGET_HOURS" \
    --attn_impl flash_attention_2 \
    --max_frames 0 \
    --save_steps "${SAVE_STEPS:-100}" \
    --device cuda:0 \
    "${EXTRA_ARGS[@]}" || rc=$?

if [[ $rc -eq 85 ]]; then
    # sft_train_videoxum.py ha salvato uno stato riprendibile prima del limite
    # SLURM: si accoda lo stesso script, che partira' alla fine di questo job.
    NEXT_RESUME="$(<"$CHECKPOINT_ROOT/$(basename "$MODEL_DIR")/resume_pointer.txt")"
    echo "[INFO] budget di tempo esaurito: accodo il resume da $NEXT_RESUME"
    sbatch --dependency="afterany:$SLURM_JOB_ID" \
        --export="ALL,SFT_RESUME=$NEXT_RESUME" \
        "$PROJECT_DIR/script/train_sft.sh"
    exit 0
elif [[ $rc -ne 0 ]]; then
    exit "$rc"
fi

echo "[INFO] SFT completata: $(date)"