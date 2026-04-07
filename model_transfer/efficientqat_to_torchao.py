import json
from pathlib import Path

import torch
from accelerate import init_empty_weights

from hf_compat import build_model_from_config, load_auto_config, resolve_hf_token
from quantize.config import maybe_load_quant_config
from quantize.torchao_adapter import build_torchao_manifest_entry
from quantize.utils import get_named_linears


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="EfficientQAT checkpoint or model directory")
    parser.add_argument("--save_dir", type=str, required=True, help="directory for TorchAO/ExecuTorch metadata output")
    parser.add_argument("--quant_config", type=str, default=None, help="optional JSON quantization config; defaults to the metadata in --model")
    parser.add_argument("--wbits", type=int, default=4, help="fallback bitwidth when no metadata is available")
    parser.add_argument("--group_size", type=int, default=128, help="fallback group size when no metadata is available")
    parser.add_argument("--trust_remote_code", action="store_true", help="enable trust_remote_code for custom model repositories")
    parser.add_argument("--token", type=str, default=None, help="HF token for gated/private models")
    args = parser.parse_args()

    hf_token = resolve_hf_token(token=args.token)
    quant_config = maybe_load_quant_config(args.quant_config or args.model, default_bits=args.wbits, default_group_size=args.group_size)
    config = load_auto_config(args.model, trust_remote_code=args.trust_remote_code, token=hf_token)
    with init_empty_weights():
        model = build_model_from_config(config=config, trust_remote_code=args.trust_remote_code)

    manifest = {
        "schema_version": 1,
        "source": "EfficientQAT",
        "backend": "TorchAO/ExecuTorch",
        "default": quant_config.default.to_dict(),
        "modules": [],
    }

    for layer_index, layer in enumerate(model.model.layers):
        for name, module in get_named_linears(layer, torch.nn.Linear):
            spec = quant_config.resolve(name, module.in_features)
            manifest["modules"].append(
                build_torchao_manifest_entry(
                    module_name=f"model.layers.{layer_index}.{name}",
                    in_features=module.in_features,
                    out_features=module.out_features,
                    spec=spec,
                )
            )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    quant_config.save(save_dir)
    manifest_path = save_dir / "torchao_quant_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Saved TorchAO manifest to {manifest_path}")


if __name__ == "__main__":
    main()
