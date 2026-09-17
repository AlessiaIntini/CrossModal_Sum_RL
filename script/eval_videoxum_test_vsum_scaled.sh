#!/bin/bash
#SBATCH --account=IscrC_HydroFM
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=eval-test-vsum-scaled
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --chdir=/leonardo/home/userexternal/aintini0/GRPOVideo
#SBATCH --output=/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/logs/eval-test-vsum-scaled-%j.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/logs/eval-test-vsum-scaled-%j.err

set -Eeuo pipefail

PROJECT_DIR="/leonardo/home/userexternal/aintini0/GRPOVideo"
VENV_DIR="${VENV_DIR:-/leonardo_scratch/fast/IscrC_MSMU/aintini0/venvs/grpovideo}"
PREDICTIONS="${PREDICTIONS:-/leonardo_work/IscrC_MSMU/aintini0/V2Xum-LLM/results/videoxum_test_vsum}"
ANNOTATIONS="${ANNOTATIONS:-/leonardo_work/IscrC_HydroFM/aintini0/datasets/videoxum/metadata/test_videoxum.json}"
OUTPUT="${OUTPUT:-/leonardo_work/IscrC_HydroFM/aintini0/GRPOVideo/results/evaluation/videoxum_test_vsum_scaled_metrics.jsonl}"
# La baseline usa posizioni normalizzate nella scala [0, 100].
BASELINE_MAX="${BASELINE_MAX:-100}"

module load python/3.11.7
source "$VENV_DIR/bin/activate"
cd "$PROJECT_DIR"

[[ -x "$VENV_DIR/bin/python" ]] || { echo "Virtualenv assente: $VENV_DIR" >&2; exit 2; }
[[ -e "$PREDICTIONS" ]] || { echo "Predizioni assenti: $PREDICTIONS" >&2; exit 2; }
[[ -f "$ANNOTATIONS" ]] || { echo "Annotazioni assenti: $ANNOTATIONS" >&2; exit 2; }
[[ "$BASELINE_MAX" =~ ^[0-9]+$ ]] && (( BASELINE_MAX > 0 )) || {
  echo "BASELINE_MAX deve essere un intero positivo" >&2
  exit 2
}

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/vsum-scaled.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT
SCALED_PREDICTIONS="$WORK_DIR/predictions"
mkdir -p "$SCALED_PREDICTIONS"
mkdir -p "$(dirname "$OUTPUT")"

# Conversione esterna: gli shard originali non vengono modificati.
python - "$PREDICTIONS" "$ANNOTATIONS" "$SCALED_PREDICTIONS" "$BASELINE_MAX" <<'PY'
import json
import re
import sys
from pathlib import Path

predictions_path = Path(sys.argv[1])
annotations_path = Path(sys.argv[2])
output_dir = Path(sys.argv[3])
baseline_max = int(sys.argv[4])

with annotations_path.open(encoding="utf-8") as handle:
    annotations = json.load(handle)

files = sorted(predictions_path.glob("*.jsonl")) if predictions_path.is_dir() else [predictions_path]
if not files:
    raise FileNotFoundError(f"Nessun JSONL trovato in {predictions_path}")

integer_re = re.compile(r"(?<![A-Za-z0-9])\d+(?![A-Za-z0-9])")
converted = 0


def scale_index(value, frame_count):
    value = int(value)
    if not 0 <= value <= baseline_max:
        return value
    return round(value * (frame_count - 1) / baseline_max)

for source_path in files:
    destination_path = output_dir / source_path.name
    with source_path.open(encoding="utf-8") as source, destination_path.open(
        "w", encoding="utf-8"
    ) as target:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            video_id = str(row.get("video_id", ""))
            if row.get("record_type") in (None, "sample") and video_id:
                if video_id not in annotations:
                    raise KeyError(
                        f"{source_path}:{line_number}: video_id {video_id!r} "
                        "non presente nelle annotazioni"
                    )
                frame_count = int(annotations[video_id]["sampled_frames"])

                if "selected_frame_indices" in row:
                    row["selected_frame_indices"] = sorted({
                        scale_index(index, frame_count)
                        for index in row["selected_frame_indices"]
                    })
                elif "summary" in row:
                    # evalVtV estrae gli interi dalla stringa summary; manteniamo
                    # la stessa struttura ma convertiamo la scala della baseline.
                    row["summary"] = integer_re.sub(
                        lambda match: str(scale_index(match.group(0), frame_count)),
                        str(row["summary"]),
                    )
                converted += 1
            target.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"[INFO] Record convertiti: {converted}")
print(f"[INFO] Predizioni convertite: {output_dir}")
PY

echo "[INFO] Valutazione con indici zero-based sulla timeline reale..."
srun python -u evaluation/evalVtV.py \
  --predictions "$SCALED_PREDICTIONS" \
  --annotations "$ANNOTATIONS" \
  --output "$OUTPUT" \
  --index-base 0

echo "[DONE] Metriche salvate in: $OUTPUT"
