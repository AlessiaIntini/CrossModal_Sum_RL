#!/bin/bash
#SBATCH --account=IscrC_HydroFM
#SBATCH --job-name=blip-embed-videoxum
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alessia.intini@polito.it
#SBATCH --partition=boost_usr_prod
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --chdir=/leonardo/home/userexternal/aintini0/GRPOVideo
#SBATCH --output=/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/logs/blip-embed-%j.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/logs/blip-embed-%j.err

# Tutto (stdout + stderr) finisce nello STESSO file .log: cosi' vedi anche gli
# errori Python/traceback senza dover aprire il .err separato.
exec 2>&1

# -E: i trap ERR valgono anche nelle funzioni/subshell; -o pipefail: cattura errori
# in una pipe. Il trap stampa la riga e il codice di uscita del comando fallito.
set -Eeuo pipefail
trap 'rc=$?; echo "[ERROR] fallito alla riga $LINENO (exit=$rc): $BASH_COMMAND"' ERR

echo "==================================================================="
echo "[INFO] Inizio script SLURM: $(date)"
echo "[INFO] Host: $(hostname)   JobID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] GPU assegnate: ${CUDA_VISIBLE_DEVICES:-N/A} (gres: ${SLURM_JOB_GPUS:-N/A})"
echo "==================================================================="

PROJECT_DIR="/leonardo/home/userexternal/aintini0/GRPOVideo"
VENV_DIR="${VENV_DIR:-/leonardo_scratch/fast/IscrC_MSMU/aintini0/venvs/grpovideo}"
DATA_ROOT="${DATA_ROOT:-/leonardo_work/IscrC_HydroFM/aintini0/datasets/videoxum}"
OUT_DIR="${OUT_DIR:-$DATA_ROOT/feat/blip}"
SCRATCH_ROOT="${SCRATCH:-/leonardo_scratch/fast/IscrC_HydroFM/aintini0}"
MODEL_DIR="${MODEL_DIR:-/leonardo_work/IscrC_HydroFM/aintini0/models/blip-image-captioning-base}"

[[ -x "$VENV_DIR/bin/python" ]] || {
  echo "[ERROR] Virtualenv assente o non eseguibile: $VENV_DIR" >&2
  exit 2
}

module load python/3.11.7
source "$VENV_DIR/bin/activate"

export MKL_THREADING_LAYER=GNU
export HF_HOME="${HF_HOME:-$SCRATCH_ROOT/GRPOVideo/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$SCRATCH_ROOT/GRPOVideo/cache/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SCRATCH_ROOT/GRPOVideo/cache/pip}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

mkdir -p "$OUT_DIR" "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR"

echo "[INFO] virtualenv: $VENV_DIR"
echo "[INFO] python    : $(which python)"
python -V
echo "[INFO] nvidia-smi:"
nvidia-smi || echo "[WARN] nvidia-smi non disponibile"

# ── path assoluti ────────────────────────────────────────────────
CODE_DIR="$PROJECT_DIR"

# Per default vengono elaborati entrambi gli split. Per limitarne l'esecuzione:
#   sbatch --export=ALL,SPLIT=train run_blip_embeddings_videoxum.sh
#   sbatch --export=ALL,SPLIT=val   run_blip_embeddings_videoxum.sh
# SPLITS permette anche una selezione esplicita, per esempio SPLITS="train val".
if [ -n "${SPLIT:-}" ]; then
  SPLITS="$SPLIT"
else
  SPLITS="${SPLITS:-train val}"
fi
MODEL_NAME="$MODEL_DIR"
SAMPLE_FPS=1.0
BATCH_SIZE=32

mkdir -p "$CODE_DIR/logs" "$OUT_DIR"

# ── pre-flight: verifica che i path esistano PRIMA di caricare il modello ──
echo "[CHECK] Verifica path di input..."
fail=0
for split in $SPLITS; do
  case "$split" in
    train|val) ;;
    *)
      echo "  [ERRORE] split non supportato: $split (ammessi: train, val)"
      fail=1
      continue
      ;;
  esac
  p="$DATA_ROOT/$split"
  if [ -d "$p" ]; then echo "  [OK] dir  $p"; else echo "  [MANCA] dir  $p"; fail=1; fi
done
if [ -f "$CODE_DIR/configuration/extract_blip_embeddings.py" ]; then
  echo "  [OK] script $CODE_DIR/configuration/extract_blip_embeddings.py"
else
  echo "  [MANCA] script $CODE_DIR/configuration/extract_blip_embeddings.py"
  fail=1
fi
if [ -f "$MODEL_DIR/config.json" ]; then
  echo "  [OK] modello locale $MODEL_DIR"
else
  echo "  [MANCA] modello locale $MODEL_DIR"
  echo "  [INFO] eseguire prima: sbatch configuration/download_blip.sh"
  fail=1
fi
if [ "$fail" -ne 0 ]; then
  echo "[ERROR] Uno o piu' path di input non esistono: interrompo prima di caricare il modello."
  exit 1
fi

# cd necessario: lo script legge/salva percorsi relativi alla cartella del progetto
# e in caso vada esteso con moduli locali (es. `import config`) li trova qui.
cd "$CODE_DIR"

echo "[INFO] data root : $DATA_ROOT"
echo "[INFO] output dir: $OUT_DIR"
echo "[INFO] model     : $MODEL_NAME"
srun python -c 'import cv2, numpy, torch, transformers; print(f"[INFO] torch={torch.__version__} cuda={torch.cuda.is_available()} transformers={transformers.__version__}"); raise SystemExit(0 if torch.cuda.is_available() else "CUDA non disponibile")'

# ── estrazione embedding (niente vLLM: modello caricato in-process via transformers) ──
# python -u = output non bufferizzato -> i print e la progress bar (tqdm) compaiono
# subito nel log (utile se il job viene killato: vedi fin dove e' arrivato).
echo "[INFO] Split richiesti: $SPLITS"
# Disattivo il trap ERR qui: il codice di uscita di Python viene gestito a mano,
# cosi' nel log e' chiaro quale split ha causato l'eventuale errore.
trap - ERR
PY_RC=0
for split in $SPLITS; do
  VIDEO_DIR="$DATA_ROOT/$split"
  echo "-------------------------------------------------------------------"
  echo "[INFO] Avvio estrazione embedding BLIP su VideoXum/$split: $(date)"
  srun python -u "$CODE_DIR/configuration/extract_blip_embeddings.py" \
    --video_dir "$VIDEO_DIR" \
    --output_dir "$OUT_DIR" \
    --model_name "$MODEL_NAME" \
    --sample_fps "$SAMPLE_FPS" \
    --batch_size "$BATCH_SIZE" \
    --device cuda
  SPLIT_RC=$?
  if [ "$SPLIT_RC" -ne 0 ]; then
    echo "[ERROR] Estrazione dello split '$split' fallita (exit=$SPLIT_RC)."
    PY_RC=$SPLIT_RC
    break
  fi
  echo "[INFO] Split '$split' completato con successo."
done

echo "==================================================================="
if [ "$PY_RC" -eq 0 ]; then
  echo "[INFO] Estrazione completata con successo per: $SPLITS (exit=0)."
  echo "[INFO] Embedding salvati in: $OUT_DIR"
else
  echo "[ERROR] Estrazione terminata con errore (exit=$PY_RC). Cerca il traceback qui sopra."
fi
echo "[INFO] Fine script SLURM: $(date)"
echo "==================================================================="
exit $PY_RC
