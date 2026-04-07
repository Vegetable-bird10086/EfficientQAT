from __future__ import annotations

from typing import Dict

from quantize.config import QuantizationSpec


def torchao_weight_dtype(bits: int) -> str:
    return f"torch.int{bits}"


def torchao_granularity(spec: QuantizationSpec, in_features: int) -> Dict[str, object]:
    if spec.granularity == "per_channel" or spec.group_size in (-1, None):
        return {"type": "PerChannel"}
    return {"type": "PerGroup", "group_size": spec.resolved_group_size(in_features)}


def torchao_mapping_type(spec: QuantizationSpec) -> str:
    return f"MappingType.{spec.mapping.upper()}"


def torchao_packing_format() -> str:
    return "IntxPackingFormat.UNPACKED_TO_INT8"


def build_torchao_manifest_entry(module_name: str, in_features: int, out_features: int, spec: QuantizationSpec) -> Dict[str, object]:
    return {
        "module_name": module_name,
        "enabled": spec.should_quantize,
        "bits": spec.bits,
        "mapping": spec.mapping,
        "granularity": spec.granularity,
        "group_size": spec.resolved_group_size(in_features),
        "torchao": {
            "weight_dtype": torchao_weight_dtype(spec.bits),
            "weight_granularity": torchao_granularity(spec, in_features),
            "weight_mapping_type": torchao_mapping_type(spec),
            "intx_packing_format": torchao_packing_format(),
        },
        "shapes": {
            "in_features": in_features,
            "out_features": out_features,
        },
    }
