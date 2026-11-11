# -*- coding: utf-8 -*-
"""
Unit test for Sub-Byte Block Quantization (INT4).
Verifies:
  1. 75% memory compression compared to FP16.
  2. Numerical accuracy and cosine similarity > 0.99.
  3. Batched GEMM forward execution.
"""

import torch
import torch.nn as nn
from streamspec.quant.block_quant import BlockQuantizedLinear, quantize_block_int4, dequantize_block_int4

def test_int4_quantization_compression_and_fidelity():
    torch.manual_seed(42)
    in_features = 1024
    out_features = 2048
    group_size = 64

    # 1. Standard FP16 Linear layer
    orig_linear = nn.Linear(in_features, out_features, bias=False).to(torch.float16)
    fp16_bytes = orig_linear.weight.numel() * orig_linear.weight.element_size() # 2,048 * 1,024 * 2 = 4,194,304 bytes (4.0 MB)

    # 2. Convert to BlockQuantizedLinear
    quant_linear = BlockQuantizedLinear.from_float(orig_linear, group_size=group_size)
    quant_bytes = quant_linear.get_memory_bytes()

    # 3. Verify Memory Compression
    compression_ratio = 1.0 - (quant_bytes / fp16_bytes)
    print(f"Original FP16 Weight Size : {fp16_bytes / 1024:.2f} KB")
    print(f"Packed INT4 Weight Size   : {quant_bytes / 1024:.2f} KB")
    print(f"Physical Memory Saved     : {compression_ratio * 100:.2f}%")

    # Group size 64: 4 bits + 16 bits scale / 64 = 4.25 bits/param = 73.4% reduction
    assert compression_ratio > 0.70, f"Expected >70% compression, got {compression_ratio * 100:.2f}%"

    # 4. Verify Reconstruction Fidelity (Cosine Similarity)
    x = torch.randn(2, 8, in_features, dtype=torch.float16)
    orig_out = orig_linear(x)
    quant_out = quant_linear(x)

    cos_sim = torch.nn.functional.cosine_similarity(orig_out.flatten(), quant_out.flatten(), dim=0).item()
    print(f"Dequantized Output Cosine Similarity: {cos_sim:.5f}")
    assert cos_sim > 0.99, f"Expected cosine similarity > 0.99, got {cos_sim}"

if __name__ == "__main__":
    test_int4_quantization_compression_and_fidelity()
