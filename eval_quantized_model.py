import argparse
import math
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from datasets import load_dataset
from safetensors.torch import load_file as load_safetensors_file
from transformers.modeling_utils import load_sharded_checkpoint

from hf_compat import (
    build_model_from_config,
    load_auto_config,
    load_auto_model_for_causal_lm,
    load_auto_tokenizer,
    resolve_hf_token,
)
from quantize.config import load_quant_config, maybe_load_quant_config
from quantize.int_linear_real import QuantLinear
from quantize.utils import get_named_linears, set_op_by_name


DEFAULT_WIKITEXT_TEST_GLOB = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "datasets--wikitext"
    / "snapshots"
)

DEFAULT_PROMPTS = [
    "Explain why the sky appears blue in simple terms.",
    "请用中文简要解释什么是梯度下降。",
    "Write a Python function that returns the nth Fibonacci number.",
]


def resolve_quant_config(explicit_path: str | None, fallback_model_dir: str, wbits: int, group_size: int):
    if explicit_path:
        config_path = Path(explicit_path)
        if not config_path.exists():
            raise FileNotFoundError(
                f"Explicit quantization config not found: {config_path}. "
                "If you intended to pass an absolute path, make sure it starts with '/'."
            )
        return load_quant_config(str(config_path), default_bits=wbits, default_group_size=group_size)
    return maybe_load_quant_config(fallback_model_dir, default_bits=wbits, default_group_size=group_size)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Quantized checkpoint directory.")
    parser.add_argument(
        "--base_model_path",
        default=None,
        help="Optional original fp16/base model path. Recommended when untouched fp params must come from the base model.",
    )
    parser.add_argument(
        "--quant_config",
        default=None,
        help="Optional quant config path. Defaults to the metadata inside --model when available.",
    )
    parser.add_argument("--wbits", type=int, default=4, help="Fallback bitwidth when no metadata is available.")
    parser.add_argument("--group_size", type=int, default=128, help="Fallback group size when no metadata is available.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ppl_seqlen", type=int, default=2048)
    parser.add_argument("--eval_ppl", action="store_true", help="Evaluate WikiText2 test perplexity.")
    parser.add_argument(
        "--wikitext_parquet",
        default=None,
        help="Optional local WikiText2 test parquet path. If omitted, the script tries the HF cache and then load_dataset.",
    )
    parser.add_argument("--generate", action="store_true", help="Run a few deterministic generation samples.")
    parser.add_argument("--prompt", action="append", default=None, help="Custom prompt. Repeat to add multiple prompts.")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--token", default=None, help="HF token for gated/private models.")
    return parser.parse_args()


def _maybe_find_wikitext_parquet() -> Path | None:
    if not DEFAULT_WIKITEXT_TEST_GLOB.exists():
        return None
    matches = sorted(DEFAULT_WIKITEXT_TEST_GLOB.glob("*/wikitext-2-raw-v1/test-00000-of-00001.parquet"))
    if not matches:
        return None
    return matches[-1]


def load_wikitext_test_ids(tokenizer, parquet_path: str | None):
    resolved_path = Path(parquet_path) if parquet_path else _maybe_find_wikitext_parquet()
    if resolved_path is not None and resolved_path.exists():
        print(f"Loading WikiText2 test from local parquet: {resolved_path}")
        table = pq.read_table(resolved_path, columns=["text"])
        texts = [value.as_py() for value in table["text"]]
    else:
        print("Loading WikiText2 test from datasets.load_dataset(...)")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = dataset["text"]
    enc = tokenizer("\n\n".join(texts), return_tensors="pt")
    return enc.input_ids


def build_quantized_model(
    model_path: str,
    base_model_path: str | None,
    quant_config_path: str | None,
    wbits: int,
    group_size: int,
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

    if base_model_path:
        print(f"Loading base model from {base_model_path}")
        model = load_auto_model_for_causal_lm(
            base_model_path,
            device_map="cpu",
            torch_dtype=torch.float16,
            trust_remote_code=trust_remote_code,
            token=hf_token,
        )
    else:
        print("Building model from config only.")
        model = build_model_from_config(
            config=config,
            torch_dtype=torch.float16,
            trust_remote_code=trust_remote_code,
        )

    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        print(
            "Resizing embeddings to match tokenizer length:",
            model.get_input_embeddings().weight.shape[0],
            "->",
            len(tokenizer),
        )
        model.resize_token_embeddings(len(tokenizer))

    quant_config = resolve_quant_config(quant_config_path, model_path, wbits=wbits, group_size=group_size)
    replaced = 0
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("This helper currently expects decoder-only models with model.model.layers.")
    for layer_index, layer in enumerate(model.model.layers):
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
    model.tie_weights()

    model_dir = Path(model_path)
    safetensors_path = model_dir / "model.safetensors"
    if safetensors_path.exists():
        state_dict = load_safetensors_file(str(safetensors_path))
        load_info = model.load_state_dict(state_dict, strict=False)
        missing_keys = list(load_info.missing_keys)
        unexpected_keys = list(load_info.unexpected_keys)
    else:
        load_info = load_sharded_checkpoint(model, model_path, strict=False, prefer_safe=True)
        missing_keys = list(load_info.missing_keys)
        unexpected_keys = list(load_info.unexpected_keys)

    allowed_missing = {"lm_head.weight"}
    unexpected_missing = [key for key in missing_keys if key not in allowed_missing]
    if unexpected_keys:
        raise RuntimeError(f"Unexpected keys while loading quantized model: {unexpected_keys}")
    if unexpected_missing:
        raise RuntimeError(f"Missing required keys while loading quantized model: {unexpected_missing}")

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.bos_token is not None:
            tokenizer.pad_token = tokenizer.bos_token

    return model, tokenizer, replaced, missing_keys


def evaluate_ppl(model, input_ids, device: str, seqlen: int):
    model = model.to(device)
    model.eval()
    nsamples = input_ids.numel() // seqlen
    if nsamples == 0:
        raise ValueError(f"Not enough tokens for ppl_seqlen={seqlen}.")

    total_nll = 0.0
    total_tokens = 0
    for index in range(nsamples):
        batch = input_ids[:, index * seqlen : (index + 1) * seqlen].to(device)
        with torch.no_grad():
            outputs = model(batch)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        tokens = shift_labels.numel()
        total_nll += loss.item() * tokens
        total_tokens += tokens
        if (index + 1) % 20 == 0 or index + 1 == nsamples:
            print(f"chunk {index + 1}/{nsamples} loss={loss.item():.4f}")
    ppl = math.exp(total_nll / total_tokens)
    print(f"WIKITEXT2_PPL {ppl}")
    return ppl


def generate_samples(model, tokenizer, prompts: Iterable[str], device: str, max_new_tokens: int):
    model = model.to(device)
    model.eval()
    for idx, prompt in enumerate(prompts):
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc.input_ids.to(device)
        attention_mask = enc.attention_mask.to(device)
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        print(f"--- sample_{idx} ---")
        print("PROMPT:", prompt)
        print("OUT   :", tokenizer.decode(generated[0], skip_special_tokens=True))


def main():
    args = parse_args()
    if not args.eval_ppl and not args.generate:
        args.eval_ppl = True

    model, tokenizer, replaced, missing_keys = build_quantized_model(
        model_path=args.model,
        base_model_path=args.base_model_path,
        quant_config_path=args.quant_config,
        wbits=args.wbits,
        group_size=args.group_size,
        trust_remote_code=args.trust_remote_code,
        token=args.token,
    )
    print("replaced_linears=", replaced)
    print("missing_keys=", missing_keys)

    if args.eval_ppl:
        input_ids = load_wikitext_test_ids(tokenizer, args.wikitext_parquet)
        print("total_tokens=", input_ids.numel())
        print("nsamples=", input_ids.numel() // args.ppl_seqlen)
        evaluate_ppl(model, input_ids, device=args.device, seqlen=args.ppl_seqlen)

    if args.generate:
        prompts = args.prompt or DEFAULT_PROMPTS
        generate_samples(model, tokenizer, prompts, device=args.device, max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    main()
