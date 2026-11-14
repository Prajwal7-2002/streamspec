# -*- coding: utf-8 -*-
"""
Double-Buffered Asynchronous Layer Prefetcher.
Overlaps memory/disk transfer of Layer (L + 1) with GPU/CPU computation of Layer (L).
Completely hides bus latency behind tensor core execution.
"""

import time
import queue
import threading
import copy
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

class DoubleBufferedPrefetcher:
    """
    Manages two alternating layer execution buffers:
      - Active Buffer: Currently executing on GPU/device.
      - Prefetch Buffer: Background thread is transferring Layer (L + 1) into it.
    """
    def __init__(self, layer_template: nn.Module, device: str = "cpu"):
        self.device = device
        self.is_cuda = (device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"))
        
        # Buffer 0 and Buffer 1 on execution device (distinct deep copies)
        self.buffers = [
            copy.deepcopy(layer_template).to(device),
            copy.deepcopy(layer_template).to(device)
        ]
        self.active_buf_idx = 0
        
        # Asynchronous stream for CUDA (if available)
        self.copy_stream = torch.cuda.Stream() if self.is_cuda else None
        
        # Background transfer management
        self.prefetch_thread: Optional[threading.Thread] = None
        self.ready_event = threading.Event()
        self.cuda_ready_event = torch.cuda.Event() if self.is_cuda else None
        
        # Performance Telemetry
        self.total_prefetch_time_sec = 0.0
        self.total_wait_stall_sec = 0.0
        self.layers_prefetched = 0

    def start_prefetch(self, next_layer_weights: Dict[str, torch.Tensor]):
        """
        Spawns or triggers asynchronous transfer of next_layer_weights
        into the standby buffer (1 - active_buf_idx).
        """
        self.ready_event.clear()
        standby_idx = 1 - self.active_buf_idx
        standby_buffer = self.buffers[standby_idx]
        
        def _transfer():
            t0 = time.perf_counter()
            if self.is_cuda and self.copy_stream is not None:
                with torch.cuda.stream(self.copy_stream):
                    with torch.no_grad():
                        for name, param in standby_buffer.named_parameters():
                            if name in next_layer_weights:
                                param.copy_(next_layer_weights[name], non_blocking=True)
                        for name, buf in standby_buffer.named_buffers():
                            if name in next_layer_weights:
                                buf.copy_(next_layer_weights[name], non_blocking=True)
                    self.cuda_ready_event.record(self.copy_stream)
            else:
                with torch.no_grad():
                    for name, param in standby_buffer.named_parameters():
                        if name in next_layer_weights:
                            param.copy_(next_layer_weights[name], non_blocking=True)
                    for name, buf in standby_buffer.named_buffers():
                        if name in next_layer_weights:
                            buf.copy_(next_layer_weights[name], non_blocking=True)
                            
            self.total_prefetch_time_sec += (time.perf_counter() - t0)
            self.layers_prefetched += 1
            self.ready_event.set()

        self.prefetch_thread = threading.Thread(target=_transfer, daemon=True)
        self.prefetch_thread.start()

    def swap_and_get_active(self) -> nn.Module:
        """
        Synchronizes with the prefetch thread, verifies weights are ready,
        and swaps the active buffer index. Returns the ready active buffer.
        """
        t0 = time.perf_counter()
        
        # 1. Wait for host prefetch thread
        if self.prefetch_thread is not None:
            self.ready_event.wait()
            self.prefetch_thread.join()
            self.prefetch_thread = None
            
        # 2. Wait for CUDA stream synchronization if on GPU
        if self.is_cuda and self.cuda_ready_event is not None:
            torch.cuda.current_stream().wait_event(self.cuda_ready_event)
            
        stall_time = time.perf_counter() - t0
        self.total_wait_stall_sec += stall_time
        
        # 3. Swap active buffer index
        self.active_buf_idx = 1 - self.active_buf_idx
        return self.buffers[self.active_buf_idx]

    def load_initial_layer(self, layer_0_weights: Dict[str, torch.Tensor]) -> nn.Module:
        """
        Loads the initial layer 0 synchronously into active buffer before streaming loop starts.
        Resets active_buf_idx to 0.
        """
        # Ensure any leftover prefetch thread has finished
        if self.prefetch_thread is not None:
            self.ready_event.wait()
            self.prefetch_thread.join()
            self.prefetch_thread = None
            
        self.active_buf_idx = 0
        active_buffer = self.buffers[0]
        with torch.no_grad():
            for name, param in active_buffer.named_parameters():
                if name in layer_0_weights:
                    param.copy_(layer_0_weights[name], non_blocking=False)
            for name, buf in active_buffer.named_buffers():
                if name in layer_0_weights:
                    buf.copy_(layer_0_weights[name], non_blocking=False)
        return active_buffer

    @property
    def overlap_efficiency(self) -> float:
        """
        Percentage of prefetch transfer time successfully overlapped
        with tensor core execution without stalling the pipeline.
        """
        if self.total_prefetch_time_sec <= 0.0:
            return 1.0
        hidden_time = max(0.0, self.total_prefetch_time_sec - self.total_wait_stall_sec)
        return min(1.0, hidden_time / self.total_prefetch_time_sec)

    def reset_telemetry(self):
        self.total_prefetch_time_sec = 0.0
        self.total_wait_stall_sec = 0.0
        self.layers_prefetched = 0

