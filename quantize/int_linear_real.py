import math
from logging import getLogger
from pathlib import Path

import torch
import torch.nn as nn
import transformers
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
from safetensors.torch import load_file as load_safetensors_file
from tqdm import tqdm
from transformers.modeling_utils import load_sharded_checkpoint

from hf_compat import build_model_from_config, load_auto_config, load_auto_tokenizer
from hf_compat import load_auto_model_for_causal_lm
from quantize.bitpacking import pack_cols, pack_rows, pad_rows, unpack_cols, unpack_rows, unpad_rows
from quantize.config import QuantizationSpec, load_quant_config, maybe_load_quant_config
from quantize.quantizer import CLIPMIN, clamp_ste, round_ste
from quantize.utils import get_named_linears, set_op_by_name

logger = getLogger(__name__)

try:
    Conv1D = transformers.pytorch_utils.Conv1D
except Exception:
    Conv1D = getattr(transformers, "Conv1D", None)

try:
    from quantize.triton_utils.kernels import dequant_dim0 as triton_dequant_dim0
except Exception:
    triton_dequant_dim0 = None


def resolve_quant_config(explicit_path, fallback_model_dir, wbits, group_size):
    if explicit_path is not None:
        config_path = Path(explicit_path)
        if not config_path.exists():
            raise FileNotFoundError(
                f"Explicit quantization config not found: {config_path}. "
                "If you intended to pass an absolute path, make sure it starts with '/'."
            )
        return load_quant_config(str(config_path), default_bits=wbits, default_group_size=group_size)
    return maybe_load_quant_config(fallback_model_dir, default_bits=wbits, default_group_size=group_size)


class TritonModuleMixin:
    @classmethod
    def warmup(cls, model, transpose: bool = False, seqlen: int = 2048):
        return None


class QuantLinear(nn.Module, TritonModuleMixin):
    QUANT_TYPE = "triton"

    def __init__(
        self,
        bits,
        group_size,
        infeatures,
        outfeatures,
        bias,
        trainable=False,
        mapping="asymmetric",
        train_scale=True,
        train_zero_point=True,
        quant_spec: QuantizationSpec | None = None,
        **kwargs,
    ):
        super().__init__()
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.group_size = group_size if group_size != -1 else infeatures
        self.maxq = 2 ** self.bits - 1
        self.quant_spec = quant_spec or QuantizationSpec(
            bits=bits,
            group_size=group_size,
            mapping=mapping,
            train_scale=train_scale,
            train_zero_point=train_zero_point,
        )
        self.mapping = self.quant_spec.mapping
        self.trainable = trainable
        self.use_fake = False
        self.fake_transpose = False

        rows_per_int = 32 // self.bits
        num_groups = math.ceil(infeatures / self.group_size)

        self.register_buffer(
            "qweight",
            torch.zeros((math.ceil(infeatures / rows_per_int), outfeatures), dtype=torch.int32),
        )
        self.register_parameter(
            "scales",
            nn.Parameter(torch.ones((num_groups, outfeatures), dtype=torch.float16), requires_grad=train_scale),
        )
        self.register_parameter(
            "zero_points",
            nn.Parameter(torch.zeros((num_groups, outfeatures), dtype=torch.float32), requires_grad=train_zero_point),
        )
        self.register_buffer(
            "qzeros",
            torch.zeros((num_groups, math.ceil(outfeatures / rows_per_int)), dtype=torch.int32),
        )
        self.register_buffer(
            "g_idx",
            torch.tensor([i // self.group_size for i in range(infeatures)], dtype=torch.int32),
        )
        if bias:
            self.register_buffer("bias", torch.zeros((outfeatures), dtype=torch.float16))
        else:
            self.bias = None

    def post_init(self):
        return None

    def _rounded_zero_points(self) -> torch.Tensor:
        return clamp_ste(round_ste(self.zero_points), 0, self.maxq)

    def _dequantize_qweight(self) -> torch.Tensor:
        if triton_dequant_dim0 is not None and self.qweight.is_cuda:
            return triton_dequant_dim0(self.qweight, self.bits, self.maxq, self.infeatures, self.outfeatures)
        return unpack_rows(self.qweight, self.bits, self.infeatures, self.outfeatures).to(self.qweight.device)

    def _pack_zero_points(self, zero_points: torch.Tensor) -> torch.Tensor:
        return pack_cols(zero_points.clamp(0, self.maxq).round().to(torch.int64), self.bits)

    def use_fake_quantization(self, del_quant: bool = False, transpose: bool = False):
        weight = self._dequantized_weight()
        if transpose:
            self.fake_transpose = True
            weight = weight.transpose(0, 1).contiguous()
        self.register_buffer("weight", weight)
        self.use_fake = True
        if del_quant:
            del self.qweight
            del self.scales
            del self.zero_points
            del self.qzeros
            del self.g_idx

    def _dequantized_weight(self) -> torch.Tensor:
        weight = self._dequantize_qweight()
        padded_weight, padded_rows = pad_rows(weight, self.group_size)
        scales = clamp_ste(self.scales, CLIPMIN, 1e4).to(weight.dtype)
        zeros = self._rounded_zero_points().to(weight.dtype)
        dequant = (
            (padded_weight.view(-1, self.group_size, self.outfeatures) - zeros.view(-1, 1, self.outfeatures))
            * scales.view(-1, 1, self.outfeatures)
        ).reshape(padded_weight.shape[0], self.outfeatures)
        return unpad_rows(dequant, padded_rows)

    def pack(self, linear, scales, zeros, g_idx=None):
        weight = linear.weight.data.clone()
        if isinstance(linear, nn.Conv2d):
            weight = weight.flatten(1)
        if Conv1D is not None and isinstance(linear, Conv1D):
            weight = weight.t()

        g_idx = torch.tensor([i // self.group_size for i in range(self.infeatures)], dtype=torch.long)

        scales = scales.to(torch.float16)
        zeros = zeros.to(torch.float32)
        self.scales = nn.Parameter(scales, requires_grad=self.quant_spec.train_scale)
        self.zero_points = nn.Parameter(zeros, requires_grad=self.quant_spec.train_zero_point)
        self.qzeros = self._pack_zero_points(zeros).to(torch.int32)
        if linear.bias is not None:
            self.bias = linear.bias.clone().half()

        transposed = weight.t().contiguous().to(torch.float32)
        expanded_scales = scales[g_idx].to(torch.float32)
        expanded_zeros = zeros[g_idx]
        intweight = torch.round(transposed / expanded_scales + expanded_zeros).clamp(0, self.maxq)
        self.qweight = pack_rows(intweight.to(torch.int64), self.bits).to(torch.int32)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        zero_points = self.zero_points.detach()
        if zero_points.device.type != "meta":
            target_device = self.qzeros.device
            if target_device.type == "meta":
                target_device = zero_points.device
            self.qzeros = self._pack_zero_points(zero_points).to(target_device)
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination.pop(prefix + "zero_points", None)
        destination[prefix + "qzeros"] = self.qzeros if keep_vars else self.qzeros.detach()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        zero_key = prefix + "zero_points"
        qzero_key = prefix + "qzeros"
        if zero_key not in state_dict and qzero_key in state_dict:
            unpacked = unpack_cols(state_dict[qzero_key], self.bits, math.ceil(self.infeatures / self.group_size), self.outfeatures)
            state_dict[zero_key] = unpacked.to(self.zero_points.dtype)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        if self.use_fake:
            weight = self.weight
            if self.fake_transpose:
                weight = weight.transpose(0, 1)
        else:
            weight = self._dequantized_weight()
        out = torch.matmul(x, weight.to(x.dtype))
        if self.bias is not None:
            out = out + self.bias
        return out


def load_quantized_model(
    model_path,
    wbits,
    group_size,
    quant_config_path=None,
    base_model_path=None,
    trust_remote_code=False,
    token=None,
):
    print(f"Loading quantized model from {model_path}")

    tokenizer = load_auto_tokenizer(
        model_path,
        use_fast=False,
        trust_remote_code=trust_remote_code,
        token=token,
    )
    config = load_auto_config(model_path, trust_remote_code=trust_remote_code, token=token)
    quant_config = resolve_quant_config(quant_config_path, model_path, wbits=wbits, group_size=group_size)

    def _prepare_quantized_modules(target_model):
        layers = target_model.model.layers
        for index in tqdm(range(len(layers))):
            layer = layers[index]
            named_linears = get_named_linears(layer, torch.nn.Linear)
            for name, module in named_linears.items():
                resolved_name = f"model.layers.{index}.{name}"
                spec = quant_config.resolve(resolved_name, module.in_features)
                if not spec.should_quantize:
                    continue
                q_linear = QuantLinear(
                    spec.bits,
                    spec.resolved_group_size(module.in_features),
                    module.in_features,
                    module.out_features,
                    not module.bias is None,
                    mapping=spec.mapping,
                    train_scale=spec.train_scale,
                    train_zero_point=spec.train_zero_point,
                    quant_spec=spec,
                )
                try:
                    q_linear.to(next(layer.parameters()).device)
                except StopIteration:
                    pass
                set_op_by_name(layer, name, q_linear)
        target_model.tie_weights()
        return target_model

    try:
        with init_empty_weights():
            model = build_model_from_config(
                config=config,
                torch_dtype=torch.float16,
                trust_remote_code=trust_remote_code,
            )
        model = _prepare_quantized_modules(model)
        device_map = infer_auto_device_map(model)
        print("Loading pre-computed quantized weights...")
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=model_path,
            device_map=device_map,
            offload_state_dict=True,
        )
    except Exception as exc:
        logger.warning("Meta-device quantized load failed, falling back to direct CPU load: %s", exc)
        if base_model_path is not None:
            model = load_auto_model_for_causal_lm(
                base_model_path,
                device_map='cpu',
                torch_dtype=torch.float16,
                trust_remote_code=trust_remote_code,
                token=token,
            )
        else:
            model = build_model_from_config(
                config=config,
                torch_dtype=torch.float16,
                trust_remote_code=trust_remote_code,
            )
        model = _prepare_quantized_modules(model)
        print("Loading pre-computed quantized weights on CPU...")
        safetensors_path = Path(model_path) / "model.safetensors"
        if safetensors_path.exists():
            state_dict = load_safetensors_file(str(safetensors_path))
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if unexpected:
                raise RuntimeError(f"Unexpected keys while loading quantized checkpoint: {unexpected}")
            allowed_missing = {"lm_head.weight"}
            unexpected_missing = [key for key in missing if key not in allowed_missing]
            if unexpected_missing:
                raise RuntimeError(f"Missing required keys while loading quantized checkpoint: {unexpected_missing}")
        else:
            load_sharded_checkpoint(model, model_path, strict=True, prefer_safe=True)

    print("Loading pre-computed quantized weights Successfully")
    return model, tokenizer


__all__ = ["QuantLinear", "load_quantized_model"]
