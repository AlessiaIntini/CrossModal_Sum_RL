"""Loading of local vision-language models and of the trained LoRA adapters."""

import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

TRAINABLE_MODEL_TYPES = {"qwen2_5_vl", "qwen3_vl"}
ZERO_SHOT_MODEL_TYPES = TRAINABLE_MODEL_TYPES | {"internvl", "video_llama_3"}
GRPO_INIT_NAME = "grpo_init.json"


def get_model_type(model_path, supported_types=TRAINABLE_MODEL_TYPES):
    # Read from the config alone, without loading the weights.
    model_type = AutoConfig.from_pretrained(model_path, local_files_only=True).model_type
    if model_type not in supported_types:
        raise ValueError(f"unsupported vision-language model: model_type={model_type!r}. "
                         f"Supported: {', '.join(sorted(supported_types))}")
    return model_type


def load_model(model_path, *, dtype, attn_implementation, supported_types=TRAINABLE_MODEL_TYPES):
    model_type = get_model_type(model_path, supported_types)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, local_files_only=True, dtype=dtype, attn_implementation=attn_implementation,
    )
    return model, model_type


def write_grpo_init(checkpoint, init_from):
    # GRPO trains a new adapter on top of the merged starting adapter: the starting adapter is
    # recorded so that inference can rebuild the same model.
    (Path(checkpoint) / GRPO_INIT_NAME).write_text(
        json.dumps({"init_from": str(init_from) if init_from else None}, indent=2) + "\n", encoding="utf-8",
    )


def read_grpo_init(checkpoint):
    path = Path(checkpoint) / GRPO_INIT_NAME
    if not path.is_file():
        # Older checkpoints have no grpo_init.json: found=False.
        return False, None
    init_from = json.loads(path.read_text(encoding="utf-8"))["init_from"]
    return True, Path(init_from) if init_from else None


def load_checkpoint_model(model_path, checkpoint, device, attn_implementation, init_from=None):
    # The starting adapter comes from grpo_init.json unless init_from is given; SFT and DPO
    # checkpoints have none and are applied directly to the base model.
    found, saved_init = read_grpo_init(checkpoint)
    if init_from is None:
        init_from = saved_init
    elif found and saved_init is not None and Path(init_from) != saved_init:
        raise SystemExit(f"[model] {checkpoint} starts from {saved_init}, not from {init_from}")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model, model_type = load_model(model_path, dtype=dtype, attn_implementation=attn_implementation)
    model = model.to(device)
    # GRPO checkpoints: merge the starting adapter into the base weights first.
    if init_from is not None:
        model = PeftModel.from_pretrained(model, init_from, is_trainable=False,
                                          local_files_only=True).merge_and_unload()
        print(f"[model] merged starting adapter: {init_from}")
    # Then the trained adapter of the checkpoint.
    model = PeftModel.from_pretrained(model, checkpoint, is_trainable=False, local_files_only=True)
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    model.eval()
    model.config.use_cache = True
    print(f"[model] base={model_path} architecture={model_type} adapter={checkpoint}")
    return model, processor, model_type, init_from
