import math

import torch
import torch.nn as nn


CLIPMIN = 1e-4


def round_ste(x: torch.Tensor) -> torch.Tensor:
    return (x.round() - x).detach() + x


def clamp_ste(x: torch.Tensor, min_value: float, max_value: float) -> torch.Tensor:
    return (x.clamp(min_value, max_value) - x).detach() + x


def _reshape_into_groups(x: torch.Tensor, group_size: int) -> tuple[torch.Tensor, int]:
    if x.ndim != 2:
        raise ValueError("UniformAffineQuantizer currently expects 2D weight tensors")
    if group_size in (-1, None):
        group_size = x.shape[-1]
    remainder = x.shape[-1] % group_size
    if remainder == 0:
        return x.reshape(-1, group_size), 0
    padded = math.ceil(x.shape[-1] / group_size) * group_size
    pad_width = padded - x.shape[-1]
    pad = torch.zeros((x.shape[0], pad_width), dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=-1).reshape(-1, group_size), pad_width


def _restore_from_groups(x: torch.Tensor, original_shape: tuple[int, int], pad_width: int) -> torch.Tensor:
    rows, cols = original_shape
    x = x.reshape(rows, cols + pad_width)
    if pad_width:
        x = x[:, :-pad_width]
    return x


class UniformAffineQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int = 8,
        group_size=None,
        weight: torch.Tensor | None = None,
        mapping: str = "asymmetric",
        train_zero_point: bool = True,
    ):
        super().__init__()
        if not 2 <= n_bits <= 16:
            raise AssertionError("bitwidth not supported")
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = 2 ** n_bits - 1
        self.mapping = mapping.lower()
        self.enable = True

        if weight is None and group_size is None:
            raise ValueError("group_size is required when weight is not provided")

        if weight is not None:
            resolved_group_size = weight.shape[-1] if group_size in (-1, None) else group_size
        else:
            resolved_group_size = group_size
        if resolved_group_size is None or resolved_group_size <= 0:
            raise ValueError("group_size must be positive or -1")
        self.group_size = resolved_group_size

        with torch.no_grad():
            if weight is None:
                self.scale = nn.Parameter(torch.ones(1))
                self.zero_point = nn.Parameter(torch.zeros(1), requires_grad=train_zero_point)
                return

            x, _ = _reshape_into_groups(weight, self.group_size)
            xmin = x.amin(dim=-1, keepdim=True)
            xmax = x.amax(dim=-1, keepdim=True)

            if self.mapping == "symmetric":
                absmax = torch.maximum(xmax.abs(), xmin.abs())
                signed_max = max(1, 2 ** (self.n_bits - 1) - 1)
                scale = (absmax / signed_max).clamp(min=CLIPMIN, max=1e4)
                midpoint = torch.full_like(scale, 2 ** (self.n_bits - 1))
                zero_point = midpoint
            elif self.mapping == "asymmetric":
                value_range = xmax - xmin
                scale = (value_range / max(1, self.qmax - self.qmin)).clamp(min=CLIPMIN, max=1e4)
                zero_point = (self.qmin - xmin / scale).clamp(min=-1e4, max=1e4).round()
            else:
                raise ValueError(f"Unsupported mapping type: {self.mapping}")

            self.scale = nn.Parameter(scale)
            self.zero_point = nn.Parameter(zero_point, requires_grad=train_zero_point)

    def change_n_bits(self, n_bits: int) -> None:
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = int(2 ** n_bits - 1)

    def fake_quant(self, x: torch.Tensor) -> torch.Tensor:
        scale = clamp_ste(self.scale, CLIPMIN, 1e4)
        zero_point = clamp_ste(round_ste(self.zero_point), self.qmin, self.qmax)

        original_shape = x.shape
        x_grouped, pad_width = _reshape_into_groups(x, self.group_size)
        x_int = round_ste(x_grouped / scale) + zero_point
        x_int = x_int.clamp(self.qmin, self.qmax)
        x_dequant = (x_int - zero_point) * scale
        return _restore_from_groups(x_dequant, original_shape, pad_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_bits >= 16 or not self.enable:
            return x
        return self.fake_quant(x)
