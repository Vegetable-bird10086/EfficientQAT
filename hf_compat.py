from __future__ import annotations

import inspect
import json
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import torch
from safetensors.torch import load_file as load_safetensors_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def resolve_hf_token(token: Optional[str] = None, use_auth_token: Optional[bool] = None) -> Optional[Any]:
    if token:
        return token
    if use_auth_token:
        return True
    return None


def _with_common_hf_kwargs(kwargs: Dict[str, Any], trust_remote_code: bool = False, token: Optional[Any] = None) -> Dict[str, Any]:
    output = dict(kwargs)
    if trust_remote_code:
        output["trust_remote_code"] = True
    if token is not None:
        output["token"] = token
    return output


def load_auto_tokenizer(
    model_name_or_path: str,
    *,
    use_fast: bool = False,
    trust_remote_code: bool = False,
    token: Optional[Any] = None,
    legacy: Optional[bool] = None,
    **kwargs: Any,
):
    load_kwargs = _with_common_hf_kwargs(kwargs, trust_remote_code=trust_remote_code, token=token)
    load_kwargs["use_fast"] = use_fast
    if legacy is not None:
        load_kwargs["legacy"] = legacy
    try:
        return AutoTokenizer.from_pretrained(model_name_or_path, **load_kwargs)
    except TypeError as exc:
        if "legacy" in str(exc) and "legacy" in load_kwargs:
            load_kwargs.pop("legacy", None)
            return AutoTokenizer.from_pretrained(model_name_or_path, **load_kwargs)
        raise


def load_auto_config(
    model_name_or_path: str,
    *,
    trust_remote_code: bool = False,
    token: Optional[Any] = None,
    **kwargs: Any,
):
    load_kwargs = _with_common_hf_kwargs(kwargs, trust_remote_code=trust_remote_code, token=token)
    return AutoConfig.from_pretrained(model_name_or_path, **load_kwargs)


def load_auto_model_for_causal_lm(
    model_name_or_path: str,
    *,
    trust_remote_code: bool = False,
    token: Optional[Any] = None,
    **kwargs: Any,
):
    load_kwargs = _with_common_hf_kwargs(kwargs, trust_remote_code=trust_remote_code, token=token)
    return AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)


def build_model_from_config(
    config,
    *,
    trust_remote_code: bool = False,
    **kwargs: Any,
):
    load_kwargs = dict(kwargs)
    if trust_remote_code:
        load_kwargs["trust_remote_code"] = True
    return AutoModelForCausalLM.from_config(config=config, **load_kwargs)


def resolve_model_dtype(config_or_model, default: torch.dtype = torch.float16) -> torch.dtype:
    torch_dtype = getattr(config_or_model, "torch_dtype", None)
    if isinstance(torch_dtype, torch.dtype):
        return torch_dtype
    if isinstance(torch_dtype, str):
        return getattr(torch, torch_dtype, default)
    if hasattr(config_or_model, "config"):
        return resolve_model_dtype(config_or_model.config, default=default)
    return default


def infer_model_kind(config_or_model) -> Optional[str]:
    model_type = getattr(config_or_model, "model_type", None)
    if model_type is None and hasattr(config_or_model, "config"):
        return infer_model_kind(config_or_model.config)
    if model_type is None:
        return None
    model_type = str(model_type).lower()
    if "llama" in model_type:
        return "llama"
    if model_type == "qwen3":
        return "qwen3"
    return None


def resolve_model_kind(config_or_model, requested: Optional[str] = "auto") -> str:
    requested = (requested or "auto").lower()
    if requested not in {"auto", "llama", "qwen3"}:
        raise ValueError(f"Unsupported model kind: {requested}")

    inferred = infer_model_kind(config_or_model)
    if requested == "auto":
        if inferred is not None:
            return inferred
        warnings.warn(
            "Could not infer model kind from config.model_type; falling back to generic decoder-only handling.",
            stacklevel=2,
        )
        return "auto"

    if inferred is not None and inferred != requested:
        raise ValueError(
            f"Requested model kind '{requested}' does not match config.model_type='{getattr(config_or_model, 'model_type', None)}'."
        )
    return requested


def get_decoder_backbone(model):
    if hasattr(model, "model"):
        return model.model
    raise ValueError(
        f"Unsupported decoder-only model layout for {model.__class__.__name__}: missing `.model` backbone."
    )


def get_transformer_layers(model):
    backbone = get_decoder_backbone(model)
    if hasattr(backbone, "layers"):
        return backbone.layers
    raise ValueError(
        f"Unsupported decoder-only model layout for {model.__class__.__name__}: missing `.model.layers`."
    )


def get_input_embedding_module(model):
    backbone = get_decoder_backbone(model)
    if hasattr(backbone, "embed_tokens"):
        return backbone.embed_tokens
    raise ValueError(
        f"Unsupported decoder-only model layout for {model.__class__.__name__}: missing `.model.embed_tokens`."
    )


def get_final_norm_module(model):
    backbone = get_decoder_backbone(model)
    if hasattr(backbone, "norm"):
        return backbone.norm
    raise ValueError(
        f"Unsupported decoder-only model layout for {model.__class__.__name__}: missing `.model.norm`."
    )


def get_rotary_embedding_module(model):
    backbone = get_decoder_backbone(model)
    return getattr(backbone, "rotary_emb", None)


def get_output_head_module(model):
    return getattr(model, "lm_head", None)


def tokenizer_is_llama_like(tokenizer) -> bool:
    name = tokenizer.__class__.__name__.lower()
    return "llama" in name


def build_trainer_processing_kwargs(trainer_cls, processing_class) -> Dict[str, Any]:
    trainer_signature = inspect.signature(trainer_cls.__init__)
    if "processing_class" in trainer_signature.parameters:
        return {"processing_class": processing_class}
    return {"tokenizer": processing_class}


def _load_checkpoint_file(checkpoint_path: Path) -> Dict[str, Any]:
    if checkpoint_path.suffix == ".safetensors":
        return load_safetensors_file(str(checkpoint_path))
    return torch.load(str(checkpoint_path), map_location="cpu")


def _resolve_shard_paths(checkpoint: str, prefer_safe: bool = True) -> list[Path]:
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_file():
        return [checkpoint_path]

    safe_index = checkpoint_path / "model.safetensors.index.json"
    bin_index = checkpoint_path / "pytorch_model.bin.index.json"
    single_safe = checkpoint_path / "model.safetensors"
    single_bin = checkpoint_path / "pytorch_model.bin"

    if prefer_safe and safe_index.exists():
        index_path = safe_index
    elif bin_index.exists():
        index_path = bin_index
    elif safe_index.exists():
        index_path = safe_index
    elif prefer_safe and single_safe.exists():
        return [single_safe]
    elif single_bin.exists():
        return [single_bin]
    elif single_safe.exists():
        return [single_safe]
    else:
        raise FileNotFoundError(f"Could not find a checkpoint under {checkpoint_path}")

    index_data = json.loads(index_path.read_text())
    weight_map = index_data.get("weight_map", {})
    if not weight_map:
        raise ValueError(f"Checkpoint index {index_path} does not contain a weight_map")
    return [checkpoint_path / shard_name for shard_name in dict.fromkeys(weight_map.values())]


def _load_sharded_checkpoint_fallback(model, checkpoint: str, strict: bool = True, prefer_safe: bool = True):
    shard_paths = _resolve_shard_paths(checkpoint, prefer_safe=prefer_safe)
    expected_keys = set(model.state_dict().keys())
    loaded_keys = set()
    unexpected_keys = set()

    for shard_path in shard_paths:
        state_dict = _load_checkpoint_file(shard_path)
        loaded_keys.update(state_dict.keys())
        load_info = model.load_state_dict(state_dict, strict=False)
        unexpected_keys.update(load_info.unexpected_keys)
        del state_dict

    missing_keys = sorted(expected_keys - loaded_keys)
    if strict and (missing_keys or unexpected_keys):
        raise RuntimeError(
            f"Error(s) in loading checkpoint for {model.__class__.__name__}: "
            f"missing_keys={missing_keys}, unexpected_keys={sorted(unexpected_keys)}"
        )
    return SimpleNamespace(
        missing_keys=missing_keys,
        unexpected_keys=sorted(unexpected_keys),
    )


def load_sharded_checkpoint_compat(model, checkpoint: str, strict: bool = True, prefer_safe: bool = True):
    try:
        from transformers.modeling_utils import load_sharded_checkpoint as hf_load_sharded_checkpoint
    except Exception:
        hf_load_sharded_checkpoint = None

    if hf_load_sharded_checkpoint is not None:
        return hf_load_sharded_checkpoint(model, checkpoint, strict=strict, prefer_safe=prefer_safe)
    return _load_sharded_checkpoint_fallback(model, checkpoint, strict=strict, prefer_safe=prefer_safe)
