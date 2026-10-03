"""Default locations of models, data and outputs, relative to the repository root."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
VIDEOXUM_ROOT = PROJECT_ROOT / "dataset" / "aintini" / "datasets" / "videoxum"
METADATA_DIR = VIDEOXUM_ROOT / "metadata"

DEFAULT_MODEL = MODELS_DIR / "Qwen3-VL-8B-Instruct"
TRAIN_JSON = METADATA_DIR / "train_videoxum.json"
VAL_JSON = METADATA_DIR / "val_videoxum.json"
TEST_JSON = METADATA_DIR / "test_videoxum.json"
PAIRS_JSON = VIDEOXUM_ROOT / "consensus" / "train_videoxum_dpo_consensus_pairs.json"
TRAIN_VIDEO_DIR = VIDEOXUM_ROOT / "train"
VAL_VIDEO_DIR = VIDEOXUM_ROOT / "val"
TEST_VIDEO_DIR = VIDEOXUM_ROOT / "test"
BLIP_FEATURE_DIR = VIDEOXUM_ROOT / "feat" / "blip"
