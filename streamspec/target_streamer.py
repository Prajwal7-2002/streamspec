# -*- coding: utf-8 -*-
"""
Target Model Layer-by-Layer Sequential Streamer.
Supports:
  1. forward_single_token (Naive baseline: 1 disk stream pass per token)
  2. forward_chunk_gemm   (StreamSpec: 1 disk stream pass for K candidate tokens)
"""

import time
import torch
import torch.nn as nn
from .architectures.llama import ModernTransformerBlock, RMSNorm
from .memory.mmap_loader import LayerBufferPool
from .memory.prefetcher import DoubleBufferedPrefetcher

class TargetLayerStreamer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 1024,
        n_layers: int = 16,
        n_q_heads: int = 8,
        n_kv_heads: int = 2,
        d_head: int = 128,
        d_ffn: int = 2816,
        device: str = "cpu",
        use_prefetcher: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.device = device
        self.use_prefetcher = use_prefetcher

        # Static in-memory embedding and LM head
        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # In a real offload environment, layer weights reside in mmap virtual storage.
        # We store them in a simulated layer bank to measure exact streaming overhead.
        self.layer_weight_bank = []
        for i in range(n_layers):
            block = ModernTransformerBlock(d_model, n_q_heads, n_kv_heads, d_head, d_ffn)
            # Store state_dict as CPU / Host memory storage
            self.layer_weight_bank.append({k: v.clone() for k, v in block.state_dict().items()})

        # Single active execution layer buffer on execution device
        template_block = ModernTransformerBlock(d_model, n_q_heads, n_kv_heads, d_head, d_ffn)
        self.buffer_pool = LayerBufferPool(template_block, device=device)
        self.prefetcher = DoubleBufferedPrefetcher(template_block, device=device) if use_prefetcher else None

        # Metrics
        self.total_stream_passes = 0
        self.total_tokens_verified = 0

    def stream_forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: list = None,
        pos_offset: int = 0
    ) -> tuple[torch.Tensor, list]:
        """
        Executes a single streaming pass through all L layers.
        When input_ids has shape [1, K], it evaluates all K candidate tokens simultaneously (GEMM).
        """
        self.total_stream_passes += 1
        seq_len = input_ids.shape[1]
        self.total_tokens_verified += seq_len

        # 1. Embed input
        h = self.tok_embeddings(input_ids)
        new_kv_caches = []

        # 2. Sequential Layer Stream: weights transferred into buffer layer-by-layer
        if self.use_prefetcher and self.prefetcher is not None:
            active_layer = self.prefetcher.load_initial_layer(self.layer_weight_bank[0])
            for layer_idx in range(self.n_layers):
                if layer_idx + 1 < self.n_layers:
                    self.prefetcher.start_prefetch(self.layer_weight_bank[layer_idx + 1])

                layer_kv = kv_caches[layer_idx] if kv_caches else None
                h, new_kv = active_layer(h, layer_kv, pos_offset=pos_offset)
                new_kv_caches.append(new_kv)

                if layer_idx + 1 < self.n_layers:
                    active_layer = self.prefetcher.swap_and_get_active()
        else:
            for layer_idx in range(self.n_layers):
                layer_weights = self.layer_weight_bank[layer_idx]
                active_layer = self.buffer_pool.stream_layer_weights(layer_weights)

                layer_kv = kv_caches[layer_idx] if kv_caches else None
                h, new_kv = active_layer(h, layer_kv, pos_offset=pos_offset)
                new_kv_caches.append(new_kv)

        # 3. Final projection
        normed = self.final_norm(h)
        logits = self.lm_head(normed) # [B, seq_len, vocab_size]
        return logits, new_kv_caches

