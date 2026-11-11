# -*- coding: utf-8 -*-
"""
DeepSeek-Style Sparse Mixture of Experts (MoE) with Speculative Chunk Expert-Unioning.
Implements Top-K routing, shared expert isolation, and chunk-level active expert unioning.
Reduces MoE weight transfer over PCIe/SSD by capitalizing on inter-token expert overlap.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Set, Tuple, Any

class ExpertMLP(nn.Module):
    """Single Feed-Forward Expert (SwiGLU)."""
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ffn, bias=False)
        self.w_up = nn.Linear(d_model, d_ffn, bias=False)
        self.w_down = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

class DeepSeekSparseMoE(nn.Module):
    """
    DeepSeek-V3 / R1 Style Sparse Mixture-of-Experts with Active Expert Unioning.
    Features:
      - 1 Always-Active Shared Expert (preserves global context).
      - N Routed Experts with Top-K dynamic gating.
      - Speculative Chunk-level Expert Union Optimizer:
        Transfers only unique active experts for candidate chunk [K, D].
    """
    def __init__(
        self,
        d_model: int = 1024,
        d_ffn: int = 2816,
        n_routed_experts: int = 64,
        top_k: int = 4,
        n_shared_experts: int = 1
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.n_routed_experts = n_routed_experts
        self.top_k = top_k
        self.n_shared_experts = n_shared_experts

        # Router / Gating Network
        self.router = nn.Linear(d_model, n_routed_experts, bias=False)

        # 1 Always-active Shared Expert (kept permanently in VRAM)
        self.shared_expert = ExpertMLP(d_model, d_ffn * n_shared_experts)

        # Routed Experts (normally stored in NVMe / host RAM)
        self.routed_experts = nn.ModuleList([
            ExpertMLP(d_model, d_ffn) for _ in range(n_routed_experts)
        ])

    def route_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes router gating weights and selected top-k expert indices.
        x: [B, seq_len, d_model]
        Returns:
          routing_weights: [B, seq_len, top_k] (normalized softmax over top-k)
          selected_experts: [B, seq_len, top_k] (expert indices in [0, n_routed_experts - 1])
        """
        logits = self.router(x) # [B, seq_len, n_routed_experts]
        top_logits, selected_experts = torch.topk(logits, self.top_k, dim=-1)
        routing_weights = F.softmax(top_logits, dim=-1)
        return routing_weights, selected_experts

    def compute_chunk_expert_union(self, x_chunk: torch.Tensor) -> Tuple[Set[int], float]:
        """
        Calculates the unique union of active experts for a speculative candidate chunk.
        x_chunk: [1, K, d_model]
        Returns:
          active_union: set of unique expert indices to stream from disk/RAM.
          reuse_factor: theoretical I/O reduction factor (naive expert loads / unique loads).
        """
        _, selected_experts = self.route_tokens(x_chunk)
        unique_experts = set(selected_experts.view(-1).tolist())
        
        k_tokens = x_chunk.shape[1]
        naive_loads = k_tokens * self.top_k
        actual_loads = len(unique_experts)
        reuse_factor = naive_loads / max(actual_loads, 1)
        
        return unique_experts, reuse_factor

    def forward(
        self,
        x: torch.Tensor,
        active_expert_subset: Set[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Executes MoE forward pass.
        If active_expert_subset is provided, only executes through those streamed experts.
        """
        B, seq_len, D = x.shape
        flat_x = x.view(-1, D)
        
        # 1. Shared Expert always computes
        shared_out = self.shared_expert(x)

        # 2. Router gating
        weights, indices = self.route_tokens(x)
        flat_weights = weights.view(-1, self.top_k)
        flat_indices = indices.view(-1, self.top_k)

        # 3. Routed computation
        routed_out = torch.zeros_like(flat_x)
        unique_experts = set(flat_indices.view(-1).tolist())
        
        # In layer-streaming, we iterate through unique active experts
        for expert_id in unique_experts:
            if active_expert_subset is not None and expert_id not in active_expert_subset:
                continue
            
            # Find tokens assigned to this expert
            mask = (flat_indices == expert_id)
            if not mask.any():
                continue
            
            # Token index and rank of this expert
            token_idx, top_pos = torch.where(mask)
            assigned_inputs = flat_x[token_idx]
            expert_result = self.routed_experts[expert_id](assigned_inputs)
            
            scale = flat_weights[token_idx, top_pos].unsqueeze(-1)
            routed_out.index_add_(0, token_idx, expert_result * scale)

        final_out = shared_out + routed_out.view(B, seq_len, D)
        
        telemetry = {
            "unique_active_experts": len(unique_experts),
            "total_routed_experts": self.n_routed_experts,
            "expert_sparsity_ratio": 1.0 - (len(unique_experts) / self.n_routed_experts),
            "reuse_factor": (seq_len * self.top_k) / max(len(unique_experts), 1)
        }
        return final_out, telemetry
