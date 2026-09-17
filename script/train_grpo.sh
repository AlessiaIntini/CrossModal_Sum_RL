#!/bin/bash
#SBATCH --account=IscrC_HydroFM
#SBATCH --job-name=grpo-dpo-consensus
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --chdir=/leonardo/home/userexternal/aintini0/GRPOVideo
#SBATCH --output=/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/logs/grpo-dpo-sft-ConsensusDataset-%j.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/logs/grpo-dpo-sft-ConsensusDataset-%j.err

set -Eeuo pipefail
trap 'rc=$?; echo "[ERROR] train_grpo.sh: riga $LINENO, exit=$rc, comando: $BASH_COMMAND" >&2' ERR

PROJECT_DIR="/leonardo/home/userexternal/aintini0/GRPOVideo"
DATA_ROOT="${DATA_ROOT:-/leonardo_work/IscrC_HydroFM/aintini0/datasets/videoxum}"
VENV_DIR="${VENV_DIR:-/leonardo_scratch/fast/IscrC_MSMU/aintini0/venvs/grpovideo}"
MODEL_DIR="${MODEL_DIR:-/leonardo_work/IscrC_HydroFM/aintini0/models/Qwen3-VL-8B-Instruct}"
DATA_JSON="${DATA_JSON:-$DATA_ROOT/metadata/train_videoxum.json}"
VAL_JSON="${VAL_JSON:-$DATA_ROOT/metadata/val_videoxum.json}"
FEATURE_DIR="${FEATURE_DIR:-$DATA_ROOT/feat/blip}"
# EXP_NAME separa checkpoint e metriche di ogni esperimento (es. EXP_NAME=no-grounding).
EXP_SUFFIX="${EXP_NAME:+-$EXP_NAME}"
CHECKPOINT_ROOT="${GRPO_CHECKPOINT_ROOT:-/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/checkpoints-grpo-dpo-sft-consensus-pairs}"
RESULTS_ROOT="${GRPO_RESULTS_ROOT:-/leonardo_work/IscrC_HydroFM/aintini0/GRPOVideo/results/grpo-dpo-sft-consensus-pairs}"
DPO_CHECKPOINT="${DPO_CHECKPOINT:-/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/checkpoints-dpo-sft-consensus-pairs/Qwen3-VL-8B-Instruct/checkpoint-best}"
SFT_CHECKPOINT="${SFT_CHECKPOINT:-/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/checkpoints-sft-consensus-pairs/Qwen3-VL-8B-Instruct/checkpoint-best}"
# F1 resta nella reward con peso ridotto (default del codice 0.25): premia anche la
# quantita' di frame, la penalita' di lunghezza resta attiva.
W_FBETA="${W_FBETA:-0.15}"
SEMANTIC_MODEL="${SEMANTIC_MODEL:-/leonardo_work/IscrC_HydroFM/aintini0/models/all-MiniLM-L6-v2}"

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
[[ -f "$SEMANTIC_MODEL/modules.json" ]] || { echo "Modello SBERT assente o incompleto: $SEMANTIC_MODEL" >&2; exit 2; }

module load python/3.11.7
source "$VENV_DIR/bin/activate"
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_DIR/.cache/torch}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export MKL_THREADING_LAYER=GNU
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "$CHECKPOINT_ROOT" "$RESULTS_ROOT" "$HF_HOME" "$TORCH_HOME"

# Tre modalita':
#   modello base:     sbatch --export=ALL,INIT_SOURCE=base train_grpo.sh
#   warm-start SFT:   sbatch --export=ALL,INIT_SOURCE=sft,SFT_CHECKPOINT=/path/to/sft/checkpoint-best train_grpo.sh
#   warm-start DPO:   sbatch --export=ALL,INIT_SOURCE=dpo,DPO_CHECKPOINT=/path/to/dpo/checkpoint train_grpo.sh
#   resume:    aggiungere GRPO_RESUME=/path/to/grpo/checkpoint
#   subset:    aggiungere TRAIN_SAMPLES=1000,VAL_SAMPLES=200 (0 = usa tutto)
# INIT_FROM resta disponibile per indicare direttamente un adapter; ha priorita'
# solo con INIT_SOURCE=auto (il valore predefinito).
INIT_SOURCE="${INIT_SOURCE:-dpo}"
case "$INIT_SOURCE" in
    base) INIT_FROM="" ;;
    sft)
        [[ -n "${SFT_CHECKPOINT:-}" ]] || { echo "INIT_SOURCE=sft richiede SFT_CHECKPOINT" >&2; exit 2; }
        INIT_FROM="$SFT_CHECKPOINT"
        ;;
    dpo)
        [[ -n "${DPO_CHECKPOINT:-}" ]] || { echo "INIT_SOURCE=dpo richiede DPO_CHECKPOINT" >&2; exit 2; }
        INIT_FROM="$DPO_CHECKPOINT"
        ;;
    auto) INIT_FROM="${INIT_FROM:-${DPO_CHECKPOINT:-${SFT_CHECKPOINT:-}}}" ;;
    *) echo "INIT_SOURCE non valido: $INIT_SOURCE (usa base, sft, dpo oppure auto)" >&2; exit 2 ;;
esac
EXTRA_ARGS=()
if [[ -n "$INIT_FROM" ]]; then
    [[ -f "$INIT_FROM/adapter_config.json" ]] || {
        echo "Checkpoint PEFT non valido (manca adapter_config.json): $INIT_FROM" >&2
        exit 2
    }
    EXTRA_ARGS+=(--init_from "$INIT_FROM")
fi
# Ablation della reward: W_FBETA, W_DIVERSITY, W_REPRESENTATIVENESS, W_GROUNDING,
# W_REFERENCE, W_FORMAT, W_SUMMARY_LENGTH (0 spegne il termine; se non impostate valgono i default
# di grpo_rewards.DEFAULT_REWARD_WEIGHTS) e LENGTH_PENALTY=0 per spegnere la penalita'.
for pair in f_beta:W_FBETA diversity:W_DIVERSITY representativeness:W_REPRESENTATIVENESS \
            grounding:W_GROUNDING reference:W_REFERENCE format:W_FORMAT \
            summary_length:W_SUMMARY_LENGTH; do
    var="${pair#*:}"
    if [[ -n "${!var:-}" ]]; then
        EXTRA_ARGS+=(--w_"${pair%%:*}" "${!var}")
    fi
done
if [[ "${LENGTH_PENALTY:-1}" == "0" ]]; then
    EXTRA_ARGS+=(--no-length_penalty)
fi
if [[ -n "${GRPO_RESUME:-}" ]]; then
    [[ -d "$GRPO_RESUME" ]] || { echo "Checkpoint GRPO assente: $GRPO_RESUME" >&2; exit 2; }
    EXTRA_ARGS+=(--resume_from "$GRPO_RESUME")
fi
# timeline_unit, max_pixels e min_pixels si ereditano dal checkpoint di partenza
# (video_encoding.json). Impostarli solo per GRPO dal modello base.
[[ -n "${TIMELINE_UNIT:-}" ]] && EXTRA_ARGS+=(--timeline_unit "$TIMELINE_UNIT")
[[ -n "${MAX_PIXELS:-}" ]] && EXTRA_ARGS+=(--max_pixels "$MAX_PIXELS")
[[ -n "${MIN_PIXELS:-}" ]] && EXTRA_ARGS+=(--min_pixels "$MIN_PIXELS")

echo "job=${SLURM_JOB_ID:-local} node=$(hostname)"
echo "model=$MODEL_DIR"
echo "data=$DATA_ROOT"
echo "features=$FEATURE_DIR"
echo "semantic_model=$SEMANTIC_MODEL"
echo "INIT_SOURCE=$INIT_SOURCE"
echo "INIT_FROM=${INIT_FROM:-nessuno; partenza dal modello base}"
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
    --semantic_model "$SEMANTIC_MODEL" \
    --semantic_device "${SEMANTIC_DEVICE:-cpu}" \
    --epochs "${EPOCHS:-1}" \
    --accum_steps "${ACCUM_STEPS:-8}" \
    --num_generations "${NUM_GENERATIONS:-4}" \
    --lr "${LR:-1e-5}" \
    --beta "${BETA:-0.04}" \
    --max_completion_length "${MAX_COMPLETION_LENGTH:-384}" \
    --summary_style "${SUMMARY_STYLE:-reference}" \
    --train_samples "${TRAIN_SAMPLES:-2000}" \
    --val_samples "${VAL_SAMPLES:-1000}" \
    --save_steps "${SAVE_STEPS:-100}" \
    --device cuda:0 \
    --attn_impl "flash_attention_2" \
    --temperature "${TEMPERATURE:-1}" \
    "${EXTRA_ARGS[@]}"

echo "==================================================================="
echo "[INFO] GRPO completato: $(date)"
echo "==================================================================="
