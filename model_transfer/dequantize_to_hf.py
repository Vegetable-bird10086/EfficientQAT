from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file as load_safetensors_file

from hf_compat import (
    build_model_from_config,
    get_output_head_module,
    get_transformer_layers,
    load_auto_config,
    load_auto_model_for_causal_lm,
    load_auto_tokenizer,
    load_sharded_checkpoint_compat,
    resolve_hf_token,
    resolve_model_dtype,
    resolve_model_kind,
)
from quantize.config import load_quant_config, maybe_load_quant_config
from quantize.int_linear_real import QuantLinear
from quantize.utils import get_named_linears, set_op_by_name


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Quantized EfficientQAT checkpoint directory.")
    parser.add_argument(
        "--base_model_path",
        default=None,
        help="Optional original fp model directory. Recommended so untouched weights and config stay aligned.",
    )
    parser.add_argument(
        "--quant_config",
        default=None,
        help="Optional quant config path. Defaults to the metadata inside --model when available.",
    )
    parser.add_argument("--wbits", type=int, default=4, help="Fallback bitwidth when no metadata is available.")
    parser.add_argument("--group_size", type=int, default=128, help="Fallback group size when no metadata is available.")
    parser.add_argument(
        "--model_kind",
        default="auto",
        choices=["auto", "llama", "qwen3"],
        help="Model family routing hint for decoder-only architectures.",
    )
    parser.add_argument(
        "--target_dtype",
        default="auto",
        choices=["auto", "fp16", "bf16", "fp32"],
        help="Output dtype for the restored dense model.",
    )
    parser.add_argument("--save_dir", required=True, help="Directory where the dequantized HF model will be saved.")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--token", default=None, help="HF token for gated/private models.")
    parser.add_argument("--dry_run", action="store_true", help="Only validate dequantization and print summary.")
    return parser.parse_args()


def _resolve_quant_config(explicit_path: str | None, fallback_model_dir: str, wbits: int, group_size: int):
    if explicit_path:
        config_path = Path(explicit_path)
        if not config_path.exists():
            raise FileNotFoundError(
                f"Explicit quantization config not found: {config_path}. "
                "If you intended to pass an absolute path, make sure it starts with '/'."
            )
        return load_quant_config(str(config_path), default_bits=wbits, default_group_size=group_size)
    return maybe_load_quant_config(fallback_model_dir, default_bits=wbits, default_group_size=group_size)


def _load_quantized_model_on_cpu(
    model_path: str,
    base_model_path: str | None,
    quant_config_path: str | None,
    wbits: int,
    group_size: int,
    model_kind: str,
    trust_remote_code: bool,
    token: str | None,
):
    hf_token = resolve_hf_token(token=token)
    model_path = str(model_path)
    tokenizer = load_auto_tokenizer(
        model_path,
        use_fast=False,
        trust_remote_code=trust_remote_code,
        token=hf_token,
    )
    config = load_auto_config(model_path, trust_remote_code=trust_remote_code, token=hf_token)
    resolve_model_kind(config, requested=model_kind)
    model_dtype = resolve_model_dtype(config)

    if base_model_path:
        print(f"Loading base model from {base_model_path}")
        model = load_auto_model_for_causal_lm(
            base_model_path,
            device_map="cpu",
            torch_dtype=model_dtype,
            trust_remote_code=trust_remote_code,
            token=hf_token,
        )
    else:
        print("Building model from config only.")
        model = build_model_from_config(
            config=config,
            torch_dtype=model_dtype,
            trust_remote_code=trust_remote_code,
        )

    embedding_rows = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_rows:
        print("Resizing embeddings:", embedding_rows, "->", len(tokenizer))
        model.resize_token_embeddings(len(tokenizer))

    quant_config = _resolve_quant_config(quant_config_path, model_path, wbits=wbits, group_size=group_size)
    replaced = 0
    has_lm_head_override = any(rule.matches("lm_head") for rule in quant_config.overrides)
    for layer_index, layer in enumerate(get_transformer_layers(model)):
        for name, module in get_named_linears(layer, torch.nn.Linear).items():
            resolved_name = f"model.layers.{layer_index}.{name}"
            spec = quant_config.resolve(resolved_name, module.in_features)
            if not spec.should_quantize:
                continue
            q_linear = QuantLinear(
                spec.bits,
                spec.resolved_group_size(module.in_features),
                module.in_features,
                module.out_features,
                module.bias is not None,
                mapping=spec.mapping,
                train_scale=spec.train_scale,
                train_zero_point=spec.train_zero_point,
                quant_spec=spec,
            )
            set_op_by_name(layer, name, q_linear)
            replaced += 1
    output_head = get_output_head_module(model)
    model.tie_weights()
    output_head = get_output_head_module(model)
    if has_lm_head_override and isinstance(output_head, torch.nn.Linear):
        spec = quant_config.resolve("lm_head", output_head.in_features)
        q_linear = QuantLinear(
            spec.bits,
            spec.resolved_group_size(output_head.in_features),
            output_head.in_features,
            output_head.out_features,
            output_head.bias is not None,
            mapping=spec.mapping,
            train_scale=spec.train_scale,
            train_zero_point=spec.train_zero_point,
            quant_spec=spec,
        )
        model.lm_head = q_linear
        replaced += 1

    model_dir = Path(model_path)
    safetensors_path = model_dir / "model.safetensors"
    if safetensors_path.exists():
        state_dict = load_safetensors_file(str(safetensors_path))
        load_info = model.load_state_dict(state_dict, strict=False)
    else:
        load_info = load_sharded_checkpoint_compat(model, model_path, strict=False, prefer_safe=True)

    lm_head_quantized = not isinstance(get_output_head_module(model), torch.nn.Linear)
    allowed_missing = set() if lm_head_quantized else {"lm_head.weight"}
    unexpected_missing = [key for key in load_info.missing_keys if key not in allowed_missing]
    if load_info.unexpected_keys:
        raise RuntimeError(f"Unexpected keys while loading quantized model: {load_info.unexpected_keys}")
    if unexpected_missing:
        raise RuntimeError(f"Missing required keys while loading quantized model: {unexpected_missing}")

    return model, tokenizer, replaced, model_dtype


def _resolve_target_dtype(target_dtype: str, fallback_dtype: torch.dtype) -> torch.dtype:
    if target_dtype == "auto":
        return fallback_dtype
    mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    return mapping[target_dtype]


def dequantize_inplace(model: nn.Module, target_dtype: torch.dtype) -> int:
    replacements: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, QuantLinear):
            continue
        dense = nn.Linear(
            module.infeatures,
            module.outfeatures,
            bias=module.bias is not None,
            dtype=target_dtype,
        )
        with torch.no_grad():
            weight = module._dequantized_weight().transpose(0, 1).contiguous().to(dtype=target_dtype, device="cpu")
            dense.weight.copy_(weight)
            if module.bias is not None:
                dense.bias.copy_(module.bias.to(dtype=target_dtype, device="cpu"))
        replacements.append((name, dense))

    for name, dense in replacements:
        set_op_by_name(model, name, dense)

    model.tie_weights()
    return len(replacements)


def main():
    args = parse_args()
    model, tokenizer, quantized_linears, inferred_dtype = _load_quantized_model_on_cpu(
        model_path=args.model,
        base_model_path=args.base_model_path,
        quant_config_path=args.quant_config,
        wbits=args.wbits,
        group_size=args.group_size,
        model_kind=args.model_kind,
        trust_remote_code=args.trust_remote_code,
        token=args.token,
    )
    target_dtype = _resolve_target_dtype(args.target_dtype, inferred_dtype)
    print(f"loaded_quantized_linears={quantized_linears}")
    print(f"target_dtype={target_dtype}")
    restored = dequantize_inplace(model, target_dtype=target_dtype)
    print(f"restored_dense_linears={restored}")

    if args.dry_run:
        print("dry_run=True, skipping save_pretrained")
        return

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(dtype=target_dtype)
    print(f"Saving dequantized model to {save_dir}")
    model.save_pretrained(save_dir, safe_serialization=True)
    tokenizer.save_pretrained(save_dir)
    print("save_success")


if __name__ == "__main__":
    main()
