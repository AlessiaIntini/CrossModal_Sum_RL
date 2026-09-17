#!/bin/bash
#SBATCH --account=IscrC_HydroFM
#SBATCH --job-name=eval-grpo-final
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --chdir=/leonardo/home/userexternal/aintini0/GRPOVideo
#SBATCH --output=/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/logs/eval-grpo-final-v2-%j.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/logs/eval-grpo-final-v2-%j.err

set -Eeuo pipefail
trap 'rc=$?; echo "[ERROR] eval_videoxum_test_grpo_final.sh: riga $LINENO, exit=$rc, comando: $BASH_COMMAND" >&2' ERR

PROJECT_DIR="/leonardo/home/userexternal/aintini0/GRPOVideo"
VENV_DIR="${VENV_DIR:-/leonardo_scratch/fast/IscrC_MSMU/aintini0/venvs/grpovideo}"
CANDIDATES_JSON="${CANDIDATES_JSON:-/leonardo_work/IscrC_HydroFM/aintini0/GRPOVideo/results/inference-grpo-dpo-v2/grpo-final-test/predictions-test-0-4000-20260913T085624727729Z.jsonl}"
REFERENCES_JSON="${REFERENCES_JSON:-/leonardo_work/IscrC_HydroFM/aintini0/datasets/videoxum/metadata/test.json}"
VSUM_ANNOTATIONS_JSON="${VSUM_ANNOTATIONS_JSON:-/leonardo_work/IscrC_HydroFM/aintini0/datasets/videoxum/metadata/test_videoxum.json}"
OUTPUT_JSONL="${OUTPUT_JSONL:-/leonardo_work/IscrC_HydroFM/aintini0/GRPOVideo/results/evaluation/videoxum_test_grpo_final_metrics_v2.jsonl}"
VSUM_OUTPUT_JSONL="${VSUM_OUTPUT_JSONL:-/leonardo_work/IscrC_HydroFM/aintini0/GRPOVideo/results/evaluation/videoxum_test_grpo_final_vsum_metrics_v2.jsonl}"
SBERT_MODEL="${SBERT_MODEL:-/leonardo_work/IscrC_HydroFM/aintini0/models/all-MiniLM-L6-v2}"
NLTK_DATA="${NLTK_DATA:-/leonardo_work/IscrC_HydroFM/aintini0/GRPOVideo/nltk_data}"
SCRATCH_ROOT="${SCRATCH:-/leonardo_scratch/fast/IscrC_HydroFM/aintini0}"

cd "$PROJECT_DIR"
[[ -x "$VENV_DIR/bin/python" ]] || { echo "Virtualenv assente: $VENV_DIR" >&2; exit 2; }
[[ -f "$CANDIDATES_JSON" ]] || { echo "File dei candidati assente: $CANDIDATES_JSON" >&2; exit 2; }
[[ -f "$REFERENCES_JSON" ]] || { echo "Riferimenti assenti: $REFERENCES_JSON" >&2; exit 2; }
[[ -f "$VSUM_ANNOTATIONS_JSON" ]] || { echo "Annotazioni frame assenti: $VSUM_ANNOTATIONS_JSON" >&2; exit 2; }
[[ -f "$SBERT_MODEL/modules.json" ]] || { echo "Modello SBERT assente o incompleto: $SBERT_MODEL" >&2; exit 2; }
[[ -d "$NLTK_DATA/corpora/wordnet" ]] || { echo "WordNet NLTK assente: $NLTK_DATA/corpora/wordnet" >&2; exit 2; }
[[ -d "$NLTK_DATA/corpora/omw-1.4" ]] || { echo "OMW NLTK assente: $NLTK_DATA/corpora/omw-1.4" >&2; exit 2; }

module load python/3.11.7
source "$VENV_DIR/bin/activate"

export HF_HOME="${HF_HOME:-$SCRATCH_ROOT/GRPOVideo/cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-$HF_HOME/sentence_transformers}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SCRATCH_ROOT/GRPOVideo/cache/pip}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NLTK_DATA

mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$SENTENCE_TRANSFORMERS_HOME" \
  "$PIP_CACHE_DIR" "$(dirname "$OUTPUT_JSONL")" "$(dirname "$VSUM_OUTPUT_JSONL")"

echo "[INFO] Avvio valutazione GRPO final test: $(date)"
echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "[INFO] candidates=$CANDIDATES_JSON"
echo "[INFO] references=$REFERENCES_JSON"
echo "[INFO] vsum_annotations=$VSUM_ANNOTATIONS_JSON"
echo "[INFO] output=$OUTPUT_JSONL"
echo "[INFO] vsum_output=$VSUM_OUTPUT_JSONL"
echo "[INFO] sbert_model=$SBERT_MODEL"
echo "[INFO] python=$(command -v python)"
python --version

srun python -u evaluation/eval.py \
  --candidates-json "$CANDIDATES_JSON" \
  --references-json "$REFERENCES_JSON" \
  --output-jsonl "$OUTPUT_JSONL" \
  --sbert-model "$SBERT_MODEL" \
  --device cpu

echo "[INFO] Calcolo metriche sui selected_frame_indices..."
srun python -u evaluation/evalVtV.py \
  --predictions "$CANDIDATES_JSON" \
  --annotations "$VSUM_ANNOTATIONS_JSON" \
  --output "$VSUM_OUTPUT_JSONL" \
  --index-base 0

echo "[INFO] Valutazione completata: $(date)"
echo "[INFO] Metriche testuali: $OUTPUT_JSONL"
echo "[INFO] Metriche frame selection: $VSUM_OUTPUT_JSONL"
