#!/bin/bash
#SBATCH --account=IscrC_HydroFM
#SBATCH --partition=lrd_all_serial
#SBATCH --job-name=videoxum_metadata
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=tmpfs:10g
#SBATCH --output=/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/logs/videoxum-metadata-%j.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_HydroFM/aintini0/GRPOVideo/logs/videoxum-metadata-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
VIDEOXUM_DATA_ROOT="${VIDEOXUM_DATA_ROOT:-$PROJECT_DIR/dataset/aintini/datasets/videoxum}"
VIDEOXUM_METADATA_DIR="${VIDEOXUM_METADATA_DIR:-$VIDEOXUM_DATA_ROOT/metadata}"

export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PROJECT_DIR/.cache/pip}"
export VIDEOXUM_METADATA_DIR

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_EXECUTABLE="$PYTHON_BIN"
elif [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
    PYTHON_EXECUTABLE="$PROJECT_DIR/.venv/bin/python"
else
    PYTHON_EXECUTABLE="python3"
fi

cd "$PROJECT_DIR"

"$PYTHON_EXECUTABLE" - <<'PY'
import json
import os
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError as error:
    raise SystemExit(
        "Dipendenza mancante: installa huggingface_hub con "
        "`python3 -m pip install huggingface_hub`."
    ) from error

output_dir = Path(os.environ["VIDEOXUM_METADATA_DIR"])
cache_dir = Path(os.environ["HF_HOME"])
filenames = [
    "train_videoxum.json",
    "val_videoxum.json",
    "test_videoxum.json",
]

output_dir.mkdir(parents=True, exist_ok=True)
cache_dir.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id="jylins/videoxum",
    repo_type="dataset",
    allow_patterns=filenames,
    local_dir=output_dir,
    cache_dir=cache_dir,
)

required_fields = {
    "video_id",
    "duration",
    "sampled_frames",
    "timestamps",
    "tsum",
    "vsum",
    "vsum_onehot",
}

for filename in filenames:
    path = output_dir / filename
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    records = list(payload.values()) if isinstance(payload, dict) else payload
    if not records:
        raise RuntimeError(f"{filename} non contiene record")
    missing_fields = required_fields - set(records[0])
    if missing_fields:
        raise RuntimeError(f"{filename}: campi mancanti {sorted(missing_fields)}")
    print(f"{filename}: {len(records)} record, {path.stat().st_size / 1024**2:.1f} MiB")

print(f"Metadati originali salvati in: {output_dir}")
PY
