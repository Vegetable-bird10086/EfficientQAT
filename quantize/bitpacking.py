import math
from typing import Tuple

import torch


def _pack_factor(bits: int) -> int:
    if bits not in {2, 3, 4, 8}:
        raise NotImplementedError("Only 2, 3, 4, and 8 bits are supported.")
    return 32 // bits


def pack_rows(values: torch.Tensor, bits: int) -> torch.Tensor:
    values = values.to(torch.int64)
    rows, cols = values.shape
    pack_factor = _pack_factor(bits)
    packed = torch.zeros((math.ceil(rows / pack_factor), cols), dtype=torch.int64, device=values.device)
    for offset in range(pack_factor):
        source_rows = values[offset::pack_factor]
        if source_rows.numel() == 0:
            continue
        packed[: source_rows.shape[0]] |= source_rows << (bits * offset)
    return packed.to(torch.int32)


def pack_cols(values: torch.Tensor, bits: int) -> torch.Tensor:
    values = values.to(torch.int64)
    rows, cols = values.shape
    pack_factor = _pack_factor(bits)
    packed = torch.zeros((rows, math.ceil(cols / pack_factor)), dtype=torch.int64, device=values.device)
    for offset in range(pack_factor):
        source_cols = values[:, offset::pack_factor]
        if source_cols.numel() == 0:
            continue
        packed[:, : source_cols.shape[1]] |= source_cols << (bits * offset)
    return packed.to(torch.int32)


def unpack_rows(packed: torch.Tensor, bits: int, rows: int, cols: int) -> torch.Tensor:
    pack_factor = _pack_factor(bits)
    packed = packed.to(torch.int64)
    unpacked = torch.zeros((rows, cols), dtype=torch.int64, device=packed.device)
    maxq = 2 ** bits - 1
    for offset in range(pack_factor):
        if offset >= rows:
            break
        row_indices = torch.arange(offset, rows, pack_factor, device=packed.device)
        if row_indices.numel() == 0:
            continue
        packed_rows = row_indices // pack_factor
        unpacked[row_indices] = (packed[packed_rows] >> (bits * offset)) & maxq
    return unpacked.to(torch.float16)


def unpack_cols(packed: torch.Tensor, bits: int, rows: int, cols: int) -> torch.Tensor:
    pack_factor = _pack_factor(bits)
    packed = packed.to(torch.int64)
    unpacked = torch.zeros((rows, cols), dtype=torch.int64, device=packed.device)
    maxq = 2 ** bits - 1
    for offset in range(pack_factor):
        if offset >= cols:
            break
        col_indices = torch.arange(offset, cols, pack_factor, device=packed.device)
        if col_indices.numel() == 0:
            continue
        packed_cols = col_indices // pack_factor
        unpacked[:, col_indices] = (packed[:, packed_cols] >> (bits * offset)) & maxq
    return unpacked.to(torch.float16)


def pad_rows(values: torch.Tensor, group_size: int) -> Tuple[torch.Tensor, int]:
    if group_size <= 0:
        return values, 0
    remainder = values.shape[0] % group_size
    if remainder == 0:
        return values, 0
    pad_rows = group_size - remainder
    padding = torch.zeros((pad_rows, values.shape[1]), dtype=values.dtype, device=values.device)
    return torch.cat([values, padding], dim=0), pad_rows


def unpad_rows(values: torch.Tensor, padded_rows: int) -> torch.Tensor:
    if padded_rows == 0:
        return values
    return values[:-padded_rows]
