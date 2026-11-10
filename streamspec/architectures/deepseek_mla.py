# -*- coding: utf-8 -*-
"""
Multi-Head Latent Attention (MLA) Architecture for DeepSeek-V3 / DeepSeek-R1.
Compresses KV-Cache footprints by 93.3% through low-rank latent projections
and decoupled rotary position embeddings.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadLatentAttention(nn.Module):
    def __init__(
        self,
        d_model: int = 2048,
        n_heads: int = 16,
        d_head: int = 128,
        d_latent_kv: int = 512,  # Low-rank latent KV compression dimension
        d_latent_q: int = 512,   # Low-rank latent Q dimension
        d_rope: int = 64,        # Decoupled RoPE dimension
        max_seq_len: int = 4096,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_latent_kv = d_latent_kv
        self.d_latent_q = d_latent_q
        self.d_rope = d_rope
        self.scale = 1.0 / math.sqrt(d_head + d_rope)

        # 1. Query Projections (Low-Rank Compression + Decoupled RoPE)
        self.w_dq = nn.Linear(d_model, d_latent_q, bias=False)
        self.q_norm = nn.RMSNorm(d_latent_q)
        self.w_uq = nn.Linear(d_latent_q, n_heads * d_head, bias=False)
        self.w_qr = nn.Linear(d_latent_q, n_heads * d_rope, bias=False)

        # 2. Key-Value Projections (Low-Rank Compression + Decoupled RoPE)
        self.w_dkv = nn.Linear(d_model, d_latent_kv, bias=False)
        self.kv_norm = nn.RMSNorm(d_latent_kv)
        self.w_uk = nn.Linear(d_latent_kv, n_heads * d_head, bias=False)
        self.w_uv = nn.Linear(d_latent_kv, n_heads * d_head, bias=False)
        self.w_kr = nn.Linear(d_model, d_rope, bias=False)

        # 3. Output Projection
        self.w_out = nn.Linear(n_heads * d_head, d_model, bias=False)

    @staticmethod
    def apply_rope(x: torch.Tensor, pos_ids: torch.Tensor, d_rope: int) -> torch.Tensor:
        """Apply Decoupled 2D Rotary Position Embeddings."""
        # x: [Batch, SeqLen, Heads, d_rope] or [Batch, SeqLen, 1, d_rope]
        dim = x.shape[-1]
        half_dim = dim // 2
        freq_seq = torch.arange(half_dim, dtype=torch.float32, device=x.device)
        inv_freq = 1.0 / (10000.0 ** (freq_seq / half_dim))
        
        # Outer product pos * inv_freq
        t = pos_ids.float().unsqueeze(-1) * inv_freq.unsqueeze(0)  # [SeqLen, half_dim]
        freqs = torch.cat([t, t], dim=-1)                          # [SeqLen, dim]
        cos = freqs.cos().unsqueeze(0).unsqueeze(2)                # [1, SeqLen, 1, dim]
        sin = freqs.sin().unsqueeze(0).unsqueeze(2)

        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        rotated = torch.cat([-x2, x1], dim=-1)
        return (x * cos) + (rotated * sin)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: dict = None,
        pos_offset: int = 0
    ) -> tuple[torch.Tensor, dict]:
        """
        Forward pass with MLA Latent Caching.
        Instead of caching full [B, H, S, d_head] keys and values,
        MLA caches ONLY:
          1. Latent KV vector: [B, S, d_latent_kv] (e.g. 512 dimensions)
          2. Decoupled RoPE Key: [B, S, 1, d_rope] (e.g. 64 dimensions)
        This yields a 93% memory savings.
        """
        batch_size, seq_len, _ = x.shape
        pos_ids = torch.arange(pos_offset, pos_offset + seq_len, device=x.device)

        # 1. Compress Query
        q_latent = self.q_norm(self.w_dq(x))                       # [B, S, d_latent_q]
        q_content = self.w_uq(q_latent).view(batch_size, seq_len, self.n_heads, self.d_head)
        q_rope = self.w_qr(q_latent).view(batch_size, seq_len, self.n_heads, self.d_rope)
        q_rope = self.apply_rope(q_rope, pos_ids, self.d_rope)

        # 2. Compress Key-Value into Latent Vector + RoPE Key
        kv_latent = self.kv_norm(self.w_dkv(x))                    # [B, S, d_latent_kv]
        k_rope = self.w_kr(x).view(batch_size, seq_len, 1, self.d_rope)
        k_rope = self.apply_rope(k_rope, pos_ids, self.d_rope)

        # 3. Update Low-Rank KV-Cache
        if kv_cache is not None and "latent_kv" in kv_cache:
            kv_latent = torch.cat([kv_cache["latent_kv"], kv_latent], dim=1)
            k_rope = torch.cat([kv_cache["rope_k"], k_rope], dim=1)
        
        new_kv_cache = {
            "latent_kv": kv_latent.detach(),
            "rope_k": k_rope.detach()
        }

        # 4. Decompress Keys & Values from Latent Memory at Attention Time
        k_content = self.w_uk(kv_latent).view(batch_size, -1, self.n_heads, self.d_head)
        v_content = self.w_uv(kv_latent).view(batch_size, -1, self.n_heads, self.d_head)

        # Transpose for multi-head attention: [B, H, S, D]
        qc = q_content.transpose(1, 2)                             # [B, H, S_q, d_head]
        kc = k_content.transpose(1, 2)                             # [B, H, S_kv, d_head]
        vc = v_content.transpose(1, 2)                             # [B, H, S_kv, d_head]
        qr = q_rope.transpose(1, 2)                                # [B, H, S_q, d_rope]
        kr = k_rope.expand(-1, -1, self.n_heads, -1).transpose(1, 2) # [B, H, S_kv, d_rope]

        # 5. Dual-Component Attention Matrix: Content + Decoupled RoPE
        # Scores = (Q_c @ K_c^T + Q_r @ K_r^T) * scale
        scores_content = torch.matmul(qc, kc.transpose(-1, -2))
        scores_rope = torch.matmul(qr, kr.transpose(-1, -2))
        attn_scores = (scores_content + scores_rope) * self.scale

        # Causal mask if decoding/evaluating chunks
        if seq_len > 1:
            mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1)
            # Offset mask for existing KV cache
            kv_len = attn_scores.shape[-1]
            if kv_len > seq_len:
                prefix_mask = torch.zeros((seq_len, kv_len - seq_len), device=x.device)
                mask = torch.cat([prefix_mask, mask], dim=-1)
            attn_scores = attn_scores + mask.unsqueeze(0).unsqueeze(0)

        attn_weights = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weights, vc)                    # [B, H, S_q, d_head]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_heads * self.d_head)

        output = self.w_out(context)
        return output, new_kv_cache

