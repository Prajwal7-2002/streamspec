# -*- coding: utf-8 -*-
"""
StreamSpec: High-Caliber Asynchronous Speculative Layer-Streaming Engine.
Democratizing Frontier & Futuristic Reasoning Models on Consumer Silicon.
"""

__version__ = "0.1.0"
__author__ = "Prajwal7-2002"

from .engine import StreamSpecEngine, StreamSpecConfig
from .draft_worker import DraftWorker
from .target_streamer import TargetLayerStreamer
from .telemetry.monitor import StreamSpecTelemetry

__all__ = [
    "StreamSpecEngine",
    "StreamSpecConfig",
    "DraftWorker",
    "TargetLayerStreamer",
    "StreamSpecTelemetry",
]

