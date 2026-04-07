import torch
import torch.nn as nn
import torch.nn.functional as F

from quantize.config import QuantizationSpec
from quantize.quantizer import UniformAffineQuantizer


class QuantLinear(nn.Module):
    def __init__(
        self,
        org_module: nn.Linear,
        wbits: int = 4,
        group_size: int = 64,
        quant_spec: QuantizationSpec | None = None,
    ):
        super().__init__()
        self.fwd_kwargs = {}
        self.fwd_func = F.linear
        self.register_parameter("weight", org_module.weight)
        if org_module.bias is not None:
            self.register_buffer("bias", org_module.bias)
        else:
            self.bias = None
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        self.use_weight_quant = False
        self.use_temporary_parameter = False

        self.quant_spec = quant_spec or QuantizationSpec(bits=wbits, group_size=group_size)
        self.weight_quantizer = UniformAffineQuantizer(
            n_bits=self.quant_spec.bits,
            group_size=self.quant_spec.resolved_group_size(self.in_features),
            weight=org_module.weight,
            mapping=self.quant_spec.mapping,
            train_zero_point=self.quant_spec.train_zero_point,
        )
        self.weight_quantizer.zero_point.requires_grad = self.quant_spec.train_zero_point
        self.weight_quantizer.scale.requires_grad = self.quant_spec.train_scale

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias
        return self.fwd_func(input, weight, bias, **self.fwd_kwargs)

    def set_quant_state(self, weight_quant: bool = False) -> None:
        self.use_weight_quant = weight_quant
