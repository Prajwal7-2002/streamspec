# -*- coding: utf-8 -*-
"""
StreamSpec: High-Caliber Asynchronous Speculative Layer-Streaming Engine.
Democratizing Frontier & Futuristic Reasoning Models on Consumer Silicon.
"""

__version__ = "1.2.0"
__author__ = "Prajwal7-2002"

from .engine import StreamSpecEngine, StreamSpecConfig, AdaptiveChunkController
from .draft_worker import DraftWorker
from .target_streamer import TargetLayerStreamer
from .telemetry.monitor import StreamSpecTelemetry
from .memory.prefetcher import DoubleBufferedPrefetcher
from .quant.block_quant import BlockQuantizedLinear, quantize_block_int4, dequantize_block_int4
from .architectures.deepseek_mla import MultiHeadLatentAttention
from .architectures.sparse_moe import DeepSeekSparseMoE

__all__ = [
    "StreamSpecEngine",
    "StreamSpecConfig",
    "AdaptiveChunkController",
    "DraftWorker",
    "TargetLayerStreamer",
    "StreamSpecTelemetry",
    "DoubleBufferedPrefetcher",
    "BlockQuantizedLinear",
    "quantize_block_int4",
    "dequantize_block_int4",
    "MultiHeadLatentAttention",
    "DeepSeekSparseMoE",
]

