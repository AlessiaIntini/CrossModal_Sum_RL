#!/bin/bash
#SBATCH --job-name=blip-embed-videoxum
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alessia.intini@polito.it
#SBATCH --partition=gpu_a40
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/aintini/RL_VideoSum/GRPOVideo/logs/blip_embed_videoxum_%j.log
#SBATCH --error=/home/aintini/RL_VideoSum/GRPOVideo/logs/blip_embed_videoxum_%j.err

# Tutto (stdout + stderr) finisce nello STESSO file .log: cosi' vedi anche gli
# errori Python/traceback senza dover aprire il .err separato.
exec 2>&1

# -E: i trap ERR valgono anche nelle funzioni/subshell; -o pipefail: cattura errori
# in una pipe. Il trap stampa la riga e il codice di uscita del comando fallito.
set -Eo pipefail
trap 'rc=$?; echo "[ERROR] fallito alla riga $LINENO (exit=$rc): $BASH_COMMAND"' ERR

echo "==================================================================="
echo "[INFO] Inizio script SLURM: $(date)"
echo "[INFO] Host: $(hostname)   JobID: ${SLURM_JOB_ID:-N/A}"
echo "[INFO] GPU assegnate: ${CUDA_VISIBLE_DEVICES:-N/A} (gres: ${SLURM_JOB_GPUS:-N/A})"
echo "==================================================================="

# ── ambiente conda (come nel template) ───────────────────────────
source /home/aintini/miniconda3/etc/profile.d/conda.sh
conda activate myenv
export MKL_THREADING_LAYER=GNU
export PATH="/home/aintini/miniconda3/envs/myenv/bin:$PATH"

echo "[INFO] conda env : ${CONDA_DEFAULT_ENV:-N/A}"
echo "[INFO] python    : $(which python)"
python -V
echo "[INFO] nvidia-smi:"
nvidia-smi || echo "[WARN] nvidia-smi non disponibile"

# ── path assoluti ────────────────────────────────────────────────
# Cartella dove risiede extract_blip_embeddings.py (aggiorna se lo metti altrove).
CODE_DIR=/home/aintini/RL_VideoSum/GRPOVideo

# Per default vengono elaborati entrambi gli split. Per limitarne l'esecuzione:
#   sbatch --export=ALL,SPLIT=train run_blip_embeddings_videoxum.sh
#   sbatch --export=ALL,SPLIT=val   run_blip_embeddings_videoxum.sh
# SPLITS permette anche una selezione esplicita, per esempio SPLITS="train val".
if [ -n "${SPLIT:-}" ]; then
  SPLITS="$SPLIT"
else
  SPLITS="${SPLITS:-train val}"
fi
DATA_ROOT="$CODE_DIR/dataset/aintini/datasets/videoxum"

# Directory letta direttamente da train_grpo.sh per tutti gli split.
OUT_DIR="$DATA_ROOT/feat/blip"

MODEL_NAME="Salesforce/blip-image-captioning-base"
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
if [ -f "$CODE_DIR/extract_blip_embeddings.py" ]; then
  echo "  [OK] script $CODE_DIR/extract_blip_embeddings.py"
else
  echo "  [MANCA] script $CODE_DIR/extract_blip_embeddings.py"
  fail=1
fi
if [ "$fail" -ne 0 ]; then
  echo "[ERROR] Uno o piu' path di input non esistono: interrompo prima di caricare il modello."
  exit 1
fi

# cd necessario: lo script legge/salva percorsi relativi alla cartella del progetto
# e in caso vada esteso con moduli locali (es. `import config`) li trova qui.
cd "$CODE_DIR"

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
  python -u "$CODE_DIR/extract_blip_embeddings.py" \
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
