#!/bin/bash
#SBATCH --account=IscrC_HydroFM
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=clip-embed-videoxum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --chdir=/leonardo/home/userexternal/aintini0/GRPOVideo
#SBATCH --output=/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/logs/clip-embed-%j.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/logs/clip-embed-%j.err

set -Eeuo pipefail
trap 'rc=$?; echo "[ERROR] riga $LINENO, exit=$rc: $BASH_COMMAND" >&2' ERR

PROJECT_DIR="/leonardo/home/userexternal/aintini0/GRPOVideo"
VENV_DIR="${VENV_DIR:-/leonardo_scratch/fast/IscrC_MSMU/aintini0/venvs/grpovideo}"
DATA_ROOT="${DATA_ROOT:-/leonardo_work/IscrC_HydroFM/aintini0/datasets/videoxum}"
MODEL_DIR="${MODEL_DIR:-/leonardo_work/IscrC_HydroFM/aintini0/models/clip-vit-large-patch14-336}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_ROOT/feat/clip-vit-large-patch14-336}"
SPLITS="${SPLITS:-test}"
BATCH_SIZE="${BATCH_SIZE:-32}"

cd "$PROJECT_DIR"
[[ -x "$VENV_DIR/bin/python" ]] || { echo "Virtualenv assente: $VENV_DIR" >&2; exit 2; }
[[ -f "$MODEL_DIR/config.json" ]] || {
    echo "Modello CLIP assente: $MODEL_DIR" >&2
    echo "Esegui prima: sbatch configuration/download_clip.sh" >&2
    exit 2
}
module load python/3.11.7
source "$VENV_DIR/bin/activate"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
mkdir -p "$OUTPUT_DIR"

echo "[INFO] split=$SPLITS model=$MODEL_DIR output=$OUTPUT_DIR"
for split in $SPLITS; do
    case "$split" in train|val|test) ;; *) echo "Split non valido: $split" >&2; exit 2 ;; esac
    metadata="$DATA_ROOT/metadata/${split}_videoxum.json"
    video_dir="$DATA_ROOT/$split"
    [[ -f "$metadata" ]] || { echo "Metadata assenti: $metadata" >&2; exit 2; }
    [[ -d "$video_dir" ]] || { echo "Video assenti: $video_dir" >&2; exit 2; }
    srun python -u configuration/extract_clip_embeddings.py \
        --metadata-json "$metadata" \
        --video-dir "$video_dir" \
        --output-dir "$OUTPUT_DIR" \
        --model "$MODEL_DIR" \
        --batch-size "$BATCH_SIZE" \
        --device cuda \
        ${OVERWRITE:+--overwrite}
done
echo "[DONE] Estrazione CLIP completata: $(date)"
