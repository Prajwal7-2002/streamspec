# -*- coding: utf-8 -*-
"""
Modern Grouped-Query Attention (GQA) & SwiGLU Transformer Block (Llama-3 / Qwen-2.5).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight

class SwiGLUMLP(nn.Module):
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ffn, bias=False)
        self.w_up = nn.Linear(d_model, d_ffn, bias=False)
        self.w_down = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_model: int = 2048,
        n_q_heads: int = 16,
        n_kv_heads: int = 4,   # GQA: e.g. 4 KV heads for 16 Q heads (4:1 ratio)
        d_head: int = 128,
        rope_base: float = 500000.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.scale = 1.0 / math.sqrt(d_head)

        self.w_q = nn.Linear(d_model, n_q_heads * d_head, bias=False)
        self.w_k = nn.Linear(d_model, n_kv_heads * d_head, bias=False)
        self.w_v = nn.Linear(d_model, n_kv_heads * d_head, bias=False)
        self.w_out = nn.Linear(n_q_heads * d_head, d_model, bias=False)

    @staticmethod
    def apply_rope(x: torch.Tensor, pos_ids: torch.Tensor) -> torch.Tensor:
        dim = x.shape[-1]
        half_dim = dim // 2
        freqs = 1.0 / (500000.0 ** (torch.arange(half_dim, device=x.device).float() / half_dim))
        t = pos_ids.float().unsqueeze(-1) * freqs.unsqueeze(0)
        t = torch.cat([t, t], dim=-1)
        cos = t.cos().unsqueeze(0).unsqueeze(2)
        sin = t.sin().unsqueeze(0).unsqueeze(2)
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        rotated = torch.cat([-x2, x1], dim=-1)
        return (x * cos) + (rotated * sin)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple = None,
        pos_offset: int = 0
    ) -> tuple[torch.Tensor, tuple]:
        batch_size, seq_len, _ = x.shape
        pos_ids = torch.arange(pos_offset, pos_offset + seq_len, device=x.device)

        q = self.w_q(x).view(batch_size, seq_len, self.n_q_heads, self.d_head)
        k = self.w_k(x).view(batch_size, seq_len, self.n_kv_heads, self.d_head)
        v = self.w_v(x).view(batch_size, seq_len, self.n_kv_heads, self.d_head)

        q = self.apply_rope(q, pos_ids)
        k = self.apply_rope(k, pos_ids)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)

        new_kv = (k.detach(), v.detach())

        # Repeat KV heads for GQA: [B, S, H_q, D]
        n_rep = self.n_q_heads // self.n_kv_heads
        if n_rep > 1:
            k_exp = k.repeat_interleave(n_rep, dim=2)
            v_exp = v.repeat_interleave(n_rep, dim=2)
        else:
            k_exp, v_exp = k, v

        q_t = q.transpose(1, 2)
        k_t = k_exp.transpose(1, 2)
        v_t = v_exp.transpose(1, 2)

        scores = torch.matmul(q_t, k_t.transpose(-1, -2)) * self.scale

        if seq_len > 1:
            mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1)
            kv_len = scores.shape[-1]
            if kv_len > seq_len:
                mask = torch.cat([torch.zeros((seq_len, kv_len - seq_len), device=x.device), mask], dim=-1)
            scores = scores + mask.unsqueeze(0).unsqueeze(0)

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_t).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.w_out(out), new_kv

class ModernTransformerBlock(nn.Module):
    """A single clean modern Transformer Layer (RMSNorm + Attention + SwiGLU)."""
    def __init__(self, d_model: int = 2048, n_q_heads: int = 16, n_kv_heads: int = 4, d_head: int = 128, d_ffn: int = 5632):
        super().__init__()
        self.input_norm = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(d_model, n_q_heads, n_kv_heads, d_head)
        self.post_norm = RMSNorm(d_model)
        self.mlp = SwiGLUMLP(d_model, d_ffn)

    def forward(self, x: torch.Tensor, kv_cache=None, pos_offset: int = 0):
        # Pre-LN Residual: x = x + Attn(RMSNorm(x))
        normed = self.input_norm(x)
        attn_out, new_kv = self.attn(normed, kv_cache, pos_offset)
        h = x + attn_out
        # Pre-LN Residual: h = h + MLP(RMSNorm(h))
        out = h + self.mlp(self.post_norm(h))
        return out, new_kv

