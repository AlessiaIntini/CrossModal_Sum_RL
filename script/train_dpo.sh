#!/bin/bash
#SBATCH --account=IscrC_HydroFM
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=dpo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --chdir=/leonardo/home/userexternal/aintini0/GRPOVideo
#SBATCH --output=/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/logs/dpo-ConsensusDataset-%j.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/logs/dpo-ConsensusDataset-%j.err

set -Eeuo pipefail
trap 'rc=$?; echo "[ERROR] train_dpo.sh: riga $LINENO, exit=$rc, comando: $BASH_COMMAND" >&2' ERR

PROJECT_DIR="/leonardo/home/userexternal/aintini0/GRPOVideo"
VENV_DIR="/leonardo_scratch/fast/IscrC_MSMU/aintini0/venvs/grpovideo"
CHECKPOINT_ROOT="${DPO_CHECKPOINT_ROOT:-/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/checkpoints-dpo-consensus-pairs}"
RESULTS_ROOT="${DPO_RESULTS_ROOT:-/leonardo_work/IscrC_HydroFM/aintini0/GRPOVideo/results/dpo-consensus-pairs}"
DATA_JSON="${DATA_JSON:-/leonardo_work/IscrC_HydroFM/aintini0/datasets/videoxum/metadata/consensus/train_videoxum_dpo_consensus_pairs.json}"
DATA_ROOT="${DATA_ROOT:-"/leonardo_work/IscrC_HydroFM/aintini0/datasets/videoxum"}"
MODEL_DIR="${MODEL_DIR:-"/leonardo_work/IscrC_HydroFM/aintini0/models/Qwen3-VL-8B-Instruct"}"
RAW_DATA_JSON="${RAW_DATA_JSON:-$DATA_ROOT/metadata/train_videoxum.json}"
VAL_JSON="${VAL_JSON:-$DATA_ROOT/metadata/val_videoxum.json}"
SFT_CHECKPOINT="${SFT_CHECKPOINT:-/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/checkpoints-sft-consensus-pairs/Qwen3-VL-8B-Instruct/checkpoint-best}"
# timeline_unit, max_pixels e min_pixels non si passano: DPO li eredita dal
# video_encoding.json dell'SFT (o del checkpoint da riprendere). Impostarli solo
# per un DPO dal modello base; un valore diverso dal checkpoint e' un errore.
PIXEL_ARGS=()
[[ -n "${TIMELINE_UNIT:-}" ]] && PIXEL_ARGS+=(--timeline_unit "$TIMELINE_UNIT")
[[ -n "${MAX_PIXELS:-}" ]] && PIXEL_ARGS+=(--max_pixels "$MAX_PIXELS")
[[ -n "${MIN_PIXELS:-}" ]] && PIXEL_ARGS+=(--min_pixels "$MIN_PIXELS")
[[ -n "${BEST_METRIC:-}" ]] && PIXEL_ARGS+=(--best_metric "$BEST_METRIC")

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

if [[ ! -f "$DATA_JSON" ]]; then
    mkdir -p "$(dirname "$DATA_JSON")"
    echo "[INFO] Creo le preferenze DPO in $DATA_JSON"
    python -u configuration/createDPODate.py \
        --input "$RAW_DATA_JSON" \
        --output "$DATA_JSON"
fi

# Un job ripreso eredita policy, reference e codifica temporale dal proprio
# training_state.pt: --init_from va passato solo al primo job della catena
# (train_dpo.py rifiuta --init_from insieme a --resume_from).
# INIT_SOURCE=sft (default): policy e reference dall'adapter SFT.
# INIT_SOURCE=base: DPO dal modello base (reference = modello base). Non c'e'
# nessun checkpoint da cui ereditare la configurazione video, quindi
# TIMELINE_UNIT, MAX_PIXELS e MIN_PIXELS vanno passati esplicitamente e devono
# coincidere con quelli dell'SFT con cui verra' confrontato.
INIT_SOURCE="${INIT_SOURCE:-base}"
case "$INIT_SOURCE" in
    sft) INIT_FROM="${INIT_FROM:-$SFT_CHECKPOINT}" ;;
    base) INIT_FROM="" ;;
    *) echo "INIT_SOURCE non valido: $INIT_SOURCE (usa sft oppure base)" >&2; exit 2 ;;
esac
TIME_BUDGET_HOURS="${TIME_BUDGET_HOURS:-21.5}"
# DPO standard (somma delle log-probabilita'): chosen e rejected hanno quasi la
# stessa lunghezza (~21 token), mentre la media per token divide il margine per
# la lunghezza e rende beta effettivo ~0.005 (loss rimasta ~0.67 nei run
# precedenti). DPO_LENGTH_NORMALIZE=1 riproduce i run precedenti.
# lr 5e-5 adatto a LoRA (i run precedenti usavano 5e-6).
DPO_LR="${DPO_LR:-5e-5}"
DPO_LENGTH_NORMALIZE="${DPO_LENGTH_NORMALIZE:-0}"
EXTRA_ARGS=()
if [[ "$DPO_LENGTH_NORMALIZE" == "1" ]]; then
    EXTRA_ARGS+=(--length-normalize)
else
    EXTRA_ARGS+=(--no-length-normalize)
fi
if [[ -n "${DPO_RESUME:-}" ]]; then
    [[ -d "$DPO_RESUME" ]] || { echo "Checkpoint DPO assente: $DPO_RESUME" >&2; exit 2; }
    EXTRA_ARGS+=(--resume_from "$DPO_RESUME")
elif [[ "$INIT_SOURCE" == "base" ]]; then
    for var in TIMELINE_UNIT MAX_PIXELS MIN_PIXELS; do
        [[ -n "${!var:-}" ]] || { echo "INIT_SOURCE=base richiede $var esplicito (stesso valore dell'SFT)" >&2; exit 2; }
    done
else
    [[ -f "$INIT_FROM/adapter_config.json" ]] || { echo "Checkpoint SFT non valido: $INIT_FROM" >&2; exit 2; }
    EXTRA_ARGS+=(--init_from "$INIT_FROM")
fi

echo "[INFO] model=$MODEL_DIR"
echo "[INFO] train_json=$DATA_JSON"
echo "[INFO] checkpoint=$CHECKPOINT_ROOT"
echo "[INFO] results=$RESULTS_ROOT"
if [[ -n "${DPO_RESUME:-}" ]]; then
    echo "[INFO] resume=$DPO_RESUME (policy e reference dal training_state)"
else
    echo "[INFO] resume=nessuno, init_source=$INIT_SOURCE, init_from=${INIT_FROM:-modello base}"
fi
echo "[INFO] train_samples=${DPO_TRAIN_SAMPLES:-4000}"
echo "[INFO] eval_samples=${DPO_EVAL_SAMPLES:-200} eval_steps=${EVAL_STEPS:-0}"
echo "[INFO] lr=$DPO_LR beta=${DPO_BETA:-0.1} length_normalize=$DPO_LENGTH_NORMALIZE"
echo "[INFO] video_time_mode=${VIDEO_TIME_MODE:-auto} time_budget_hours=$TIME_BUDGET_HOURS"
echo "[INFO] python=$(command -v python)"
python --version
nvidia-smi
srun python -c 'import sys, torch; print(f"[INFO] torch={torch.__version__} cuda_build={torch.version.cuda} cuda_available={torch.cuda.is_available()}"); sys.exit(0 if torch.cuda.is_available() else "PyTorch non riesce a inizializzare CUDA")'

rc=0
srun python -u train_dpo.py \
    --model "$MODEL_DIR" --data_json "$DATA_JSON" \
    --video_dir "$DATA_ROOT/train" --val_json "$VAL_JSON" \
    --val_video_dir "$DATA_ROOT/val" --saveroot "$CHECKPOINT_ROOT" \
    --resultsroot "$RESULTS_ROOT" --epochs "${DPO_EPOCHS:-1}" \
    --accum_steps "${DPO_ACCUM_STEPS:-16}" --lr "$DPO_LR" \
    --beta "${DPO_BETA:-0.1}" \
    --debug_samples "${DPO_TRAIN_SAMPLES:-4000}" \
    --eval_samples "${DPO_EVAL_SAMPLES:-200}" \
    --eval_steps "${EVAL_STEPS:-0}" --eval_steps_samples "${EVAL_STEPS_SAMPLES:-200}" \
    --video_time_mode "${VIDEO_TIME_MODE:-auto}" \
    --time_budget_hours "$TIME_BUDGET_HOURS" \
    --max_frames 0 --save_steps "${SAVE_STEPS:-100}" --device cuda:0 \
    --eval_max_new_tokens "${DPO_MAX_NEW_TOKENS:-384}" \
    --attn_impl "flash_attention_2" \
    "${PIXEL_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" || rc=$?

if [[ $rc -eq 85 ]]; then
    # train_dpo.py ha salvato uno stato riprendibile prima del limite SLURM:
    # si accoda lo stesso script, che partira' alla fine di questo job.
    NEXT_RESUME="$(<"$CHECKPOINT_ROOT/$(basename "$MODEL_DIR")/resume_pointer.txt")"
    echo "[INFO] budget di tempo esaurito: accodo il resume da $NEXT_RESUME"
    sbatch --dependency="afterany:$SLURM_JOB_ID" \
        --export="ALL,DPO_RESUME=$NEXT_RESUME" \
        "$PROJECT_DIR/script/train_dpo.sh"
    exit 0
elif [[ $rc -ne 0 ]]; then
    exit "$rc"
fi

echo "==================================================================="
echo "[INFO] DPO completato: $(date)"
echo "==================================================================="
