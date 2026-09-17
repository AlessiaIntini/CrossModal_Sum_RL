#!/bin/bash
#SBATCH --account=IscrC_HydroFM
#SBATCH --job-name=eval-coco
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --chdir=/leonardo/home/userexternal/aintini0/GRPOVideo
#SBATCH --output=/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/logs/eval-coco-%j.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/logs/eval-coco-%j.err

# Metriche T-Sum con il protocollo COCO di VideoXum (confrontabili coi paper).
# Uso: CANDIDATES_JSON=<predizioni.jsonl|dir> OUTPUT_JSON=<out.json> sbatch script/eval_videoxum_test_coco.sh

set -Eeuo pipefail
trap 'rc=$?; echo "[ERROR] eval_videoxum_test_coco.sh: riga $LINENO, exit=$rc, comando: $BASH_COMMAND" >&2' ERR

PROJECT_DIR="/leonardo/home/userexternal/aintini0/GRPOVideo"
VENV_DIR="${VENV_DIR:-/leonardo_scratch/fast/IscrC_MSMU/aintini0/venvs/grpovideo}"
CANDIDATES_JSON="${CANDIDATES_JSON:?imposta CANDIDATES_JSON}"
REFERENCES_JSON="${REFERENCES_JSON:-/leonardo_work/IscrC_HydroFM/aintini0/datasets/videoxum/metadata/test.json}"
OUTPUT_JSON="${OUTPUT_JSON:?imposta OUTPUT_JSON}"
MISSING="${MISSING:-empty}"

cd "$PROJECT_DIR"
module load python/3.11.7
module load openjdk/11.0.20.1_1
source "$VENV_DIR/bin/activate"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false

echo "[INFO] candidates=$CANDIDATES_JSON"
echo "[INFO] output=$OUTPUT_JSON missing=$MISSING"
python -u evaluation/eval_coco.py \
  --candidates-json "$CANDIDATES_JSON" \
  --references-json "$REFERENCES_JSON" \
  --output-json "$OUTPUT_JSON" \
  --missing "$MISSING"
