#!/bin/bash
#SBATCH --account=IscrC_MSMU
#SBATCH --partition=lrd_all_serial
#SBATCH --job-name=videoxum_metadata
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=tmpfs:10g
#SBATCH --output=/leonardo_scratch/fast/IscrC_MSMU/aintini0/GRPOVideo/logs/videoxum-metadata-%j.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_MSMU/aintini0/GRPOVideo/logs/videoxum-metadata-%j.err

set -euo pipefail

cd /leonardo/home/userexternal/aintini0/GRPOVideo
module load python/3.11.7
source /leonardo_scratch/fast/IscrC_MSMU/aintini0/venvs/grpovideo/bin/activate

export HF_HOME="${SCRATCH}/GRPOVideo/cache/huggingface"
export PIP_CACHE_DIR="${SCRATCH}/GRPOVideo/cache/pip"

python - <<'PY'
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download

output_dir = (
    Path(os.environ["WORK"])
    / os.environ["USER"]
    / "datasets"
    / "videoxum"
    / "metadata"
    / "huggingface_original"
)
cache_dir = (
    Path(os.environ.get("SCRATCH") or os.environ["CINECA_SCRATCH"])
    / "videoxum_metadata"
    / "hf_cache"
)
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
