# -*- coding: utf-8 -*-
"""
Sub-Byte Block Quantization Engine (INT4 / FP8).
Compresses streaming layer payloads by 75% across the PCIe/SSD bus.
Enables 70B models to stream with only 35 GB of weight data.
"""

import math
import torch
import torch.nn as nn
from typing import Tuple

def quantize_block_int4(tensor: torch.Tensor, group_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes a 2D weight matrix [out_features, in_features] to signed 4-bit integers [-8, 7]
    with per-block FP16 scale factors.
    Packs two 4-bit integers per uint8 byte.
    """
    out_features, in_features = tensor.shape
    assert in_features % group_size == 0, f"in_features ({in_features}) must be divisible by group_size ({group_size})"
    
    # Reshape to [total_groups, group_size]
    reshaped = tensor.view(-1, group_size)
    
    # Compute per-block max absolute value scale
    max_val = torch.max(torch.abs(reshaped), dim=-1, keepdim=True)[0]
    scales = torch.clamp(max_val / 7.0, min=1e-6).to(torch.float16)
    
    # Quantize to [-8, 7]
    scaled = torch.round(reshaped / scales)
    q = torch.clamp(scaled, -8, 7).to(torch.int8)
    
    # Re-offset to unsigned 4-bit [0, 15] for bit packing: q_unsigned = q + 8
    q_unsigned = (q + 8).to(torch.uint8)
    
    # Pack adjacent pairs of 4-bit values into uint8 (low nibble, high nibble)
    q_flat = q_unsigned.view(-1, 2)
    packed = (q_flat[:, 0] | (q_flat[:, 1] << 4)).view(out_features, in_features // 2)
    
    scales_out = scales.view(out_features, in_features // group_size)
    return packed, scales_out

def dequantize_block_int4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 64,
    dtype: torch.dtype = torch.float16
) -> torch.Tensor:
    """
    Unpacks signed 4-bit weights from uint8 and scales them back to float16.
    """
    out_features = packed.shape[0]
    in_features = packed.shape[1] * 2
    
    # Unpack low and high nibbles
    flat_packed = packed.view(-1, 1)
    low_nibble = (flat_packed & 0x0F).to(torch.int8) - 8
    high_nibble = ((flat_packed >> 4) & 0x0F).to(torch.int8) - 8
    
    unpacked = torch.stack([low_nibble, high_nibble], dim=1).view(-1, group_size).to(dtype)
    scales_flat = scales.view(-1, 1).to(dtype)
    
    # Multiply by scale factor
    dequantized = (unpacked * scales_flat).view(out_features, in_features)
    return dequantized

class BlockQuantizedLinear(nn.Module):
    """
    A Linear layer that stores weights in compressed INT4 format (0.5 bytes/param).
    Transfers 75% fewer bytes across PCIe/SSD, and performs dequantized GEMM.
    """
    def __init__(self, in_features: int, out_features: int, group_size: int = 64, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        
        # Register packed buffers
        self.register_buffer("packed_weight", torch.zeros((out_features, in_features // 2), dtype=torch.uint8))
        self.register_buffer("scales", torch.zeros((out_features, in_features // group_size), dtype=torch.float16))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_float(cls, linear_module: nn.Linear, group_size: int = 64) -> "BlockQuantizedLinear":
        """
        Quantizes a standard FP32/FP16 nn.Linear into a BlockQuantizedLinear layer.
        """
        out_f, in_f = linear_module.weight.shape
        quant_layer = cls(in_f, out_f, group_size=group_size, bias=(linear_module.bias is not None))
        
        packed, scales = quantize_block_int4(linear_module.weight.detach(), group_size=group_size)
        quant_layer.packed_weight.copy_(packed)
        quant_layer.scales.copy_(scales)
        
        if linear_module.bias is not None:
            quant_layer.bias.data.copy_(linear_module.bias.detach().to(torch.float16))
            
        return quant_layer

    def get_memory_bytes(self) -> int:
        """Returns physical memory consumption in bytes."""
        w_bytes = self.packed_weight.numel() * self.packed_weight.element_size()
        s_bytes = self.scales.numel() * self.scales.element_size()
        b_bytes = self.bias.numel() * self.bias.element_size() if self.bias is not None else 0
        return w_bytes + s_bytes + b_bytes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs dequantized GEMM:
        x: [B, seq_len, in_features]
        out: [B, seq_len, out_features]
        """
        w_dequant = dequantize_block_int4(
            self.packed_weight,
            self.scales,
            group_size=self.group_size,
            dtype=x.dtype
        )
        return torch.nn.functional.linear(x, w_dequant, self.bias)

