import fnmatch
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_QUANT_CONFIG_FILENAME = "efficientqat_quant_config.json"


def _normalize_mapping(mapping: Optional[str]) -> str:
    value = (mapping or "asymmetric").lower()
    if value not in {"asymmetric", "symmetric"}:
        raise ValueError(f"Unsupported mapping type: {mapping}")
    return value


def _normalize_granularity(granularity: Optional[str]) -> str:
    value = (granularity or "per_group").lower()
    if value not in {"per_group", "per_channel"}:
        raise ValueError(f"Unsupported granularity: {granularity}")
    return value


@dataclass
class QuantizationSpec:
    bits: int = 4
    group_size: int = 128
    mapping: str = "asymmetric"
    granularity: str = "per_group"
    enabled: bool = True
    train_scale: bool = True
    train_zero_point: bool = True

    def __post_init__(self) -> None:
        self.mapping = _normalize_mapping(self.mapping)
        self.granularity = _normalize_granularity(self.granularity)
        if self.bits < 2 and self.enabled:
            raise ValueError(f"Unsupported bitwidth: {self.bits}")
        if self.group_size == 0:
            raise ValueError("group_size must be positive, -1, or omitted")
        if self.granularity == "per_channel":
            self.group_size = -1
        if self.mapping == "symmetric" and not self.train_zero_point:
            return
        if self.mapping == "symmetric":
            self.train_zero_point = False

    @property
    def should_quantize(self) -> bool:
        return self.enabled and self.bits < 16

    def resolved_group_size(self, in_features: int) -> int:
        if self.granularity == "per_channel" or self.group_size in (-1, None):
            return in_features
        return self.group_size

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]], fallback: Optional["QuantizationSpec"] = None) -> "QuantizationSpec":
        base = fallback.to_dict() if fallback is not None else {}
        if data:
            base.update(data)
        return cls(**base)


@dataclass
class QuantizationRule:
    pattern: str
    spec: QuantizationSpec

    def matches(self, module_name: str) -> bool:
        if fnmatch.fnmatch(module_name, self.pattern):
            return True
        return self.pattern in module_name

    def to_dict(self) -> Dict[str, Any]:
        return {"pattern": self.pattern, **self.spec.to_dict()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any], fallback: QuantizationSpec) -> "QuantizationRule":
        if "pattern" not in data:
            raise ValueError("Each quantization override must define a pattern")
        pattern = data["pattern"]
        spec_data = {k: v for k, v in data.items() if k != "pattern"}
        return cls(pattern=pattern, spec=QuantizationSpec.from_dict(spec_data, fallback=fallback))


@dataclass
class EfficientQATQuantConfig:
    default: QuantizationSpec = field(default_factory=QuantizationSpec)
    overrides: List[QuantizationRule] = field(default_factory=list)
    schema_version: int = 1

    def resolve(self, module_name: str, in_features: Optional[int] = None) -> QuantizationSpec:
        spec = QuantizationSpec.from_dict(None, fallback=self.default)
        for rule in self.overrides:
            if rule.matches(module_name):
                spec = QuantizationSpec.from_dict(rule.spec.to_dict(), fallback=spec)
        if in_features is not None and spec.group_size not in (-1, None) and spec.group_size > in_features:
            spec = QuantizationSpec.from_dict(
                {
                    "group_size": in_features,
                    "granularity": "per_channel" if spec.granularity == "per_channel" else spec.granularity,
                },
                fallback=spec,
            )
        return spec

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "default": self.default.to_dict(),
            "overrides": [rule.to_dict() for rule in self.overrides],
        }

    def save(self, output_dir: str) -> Path:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        config_path = output_path / DEFAULT_QUANT_CONFIG_FILENAME
        config_path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return config_path

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]], fallback: Optional["EfficientQATQuantConfig"] = None) -> "EfficientQATQuantConfig":
        if data is None:
            if fallback is None:
                return cls()
            return cls.from_dict(fallback.to_dict())

        base_default = fallback.default if fallback is not None else QuantizationSpec()
        default = QuantizationSpec.from_dict(data.get("default"), fallback=base_default)
        base_rules = fallback.overrides if fallback is not None else []
        overrides = list(base_rules)
        for rule_data in data.get("overrides", []):
            overrides.append(QuantizationRule.from_dict(rule_data, fallback=default))
        return cls(default=default, overrides=overrides, schema_version=data.get("schema_version", 1))


def load_quant_config(config_path: Optional[str], default_bits: int = 4, default_group_size: int = 128) -> EfficientQATQuantConfig:
    fallback = EfficientQATQuantConfig(
        default=QuantizationSpec(bits=default_bits, group_size=default_group_size),
    )
    if config_path is None:
        return fallback

    path = Path(config_path)
    if path.is_dir():
        path = path / DEFAULT_QUANT_CONFIG_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Quantization config not found: {path}")

    data = json.loads(path.read_text())
    return EfficientQATQuantConfig.from_dict(data, fallback=fallback)


def maybe_load_quant_config(model_dir: Optional[str], default_bits: int = 4, default_group_size: int = 128) -> EfficientQATQuantConfig:
    if model_dir is None:
        return load_quant_config(None, default_bits=default_bits, default_group_size=default_group_size)
    path = Path(model_dir)
    if path.is_file():
        return load_quant_config(str(path), default_bits=default_bits, default_group_size=default_group_size)
    config_path = path / DEFAULT_QUANT_CONFIG_FILENAME
    if config_path.exists():
        return load_quant_config(str(config_path), default_bits=default_bits, default_group_size=default_group_size)
    return load_quant_config(None, default_bits=default_bits, default_group_size=default_group_size)


def summarize_quantized_modules(module_specs: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    quantized = 0
    skipped = 0
    bits_histogram: Dict[str, int] = {}
    for item in module_specs:
        spec = item["spec"]
        if spec.should_quantize:
            quantized += 1
            bits_histogram[str(spec.bits)] = bits_histogram.get(str(spec.bits), 0) + 1
        else:
            skipped += 1
    return {
        "quantized_modules": quantized,
        "skipped_modules": skipped,
        "bitwidth_histogram": bits_histogram,
    }


def is_uniform_quant_config(config: EfficientQATQuantConfig) -> bool:
    return len(config.overrides) == 0
