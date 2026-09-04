"""Caricamento locale uniforme per Qwen2.5-VL e Qwen3-VL."""

from pathlib import Path

from transformers import AutoConfig, AutoModelForImageTextToText


SUPPORTED_QWEN_VL_MODEL_TYPES = {
    "qwen2_5_vl",
    "qwen3_vl",
}


def get_vl_model_type(model_path: Path) -> str:
    """Legge il tipo architetturale senza caricare i pesi del modello."""
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    model_type = config.model_type
    if model_type not in SUPPORTED_QWEN_VL_MODEL_TYPES:
        supported = ", ".join(sorted(SUPPORTED_QWEN_VL_MODEL_TYPES))
        raise ValueError(
            f"Modello vision-language non supportato: model_type={model_type!r}. "
            f"Supportati: {supported}"
        )
    return model_type


def load_qwen_vl_model(model_path: Path, *, torch_dtype, attn_implementation: str):
    """Carica automaticamente Qwen2.5-VL o Qwen3-VL dal ``config.json`` locale."""
    model_type = get_vl_model_type(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )
    return model, model_type
