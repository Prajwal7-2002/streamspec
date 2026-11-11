# -*- coding: utf-8 -*-
"""
Unit test for DeepSeek Sparse Mixture of Experts with Active Expert Unioning.
Verifies:
  1. Top-K routing dynamics and output shape.
  2. Active Expert Union calculation across speculative candidate chunk.
  3. Proven Expert Reuse Factor (saving disk/RAM loading on MoE layers).
"""

import torch
from streamspec.architectures.sparse_moe import DeepSeekSparseMoE

def test_sparse_moe_expert_unioning():
    torch.manual_seed(42)
    d_model = 256
    d_ffn = 512
    n_routed_experts = 64
    top_k = 4
    k_chunk = 8

    # 1. Instantiate DeepSeek-style MoE
    moe = DeepSeekSparseMoE(
        d_model=d_model,
        d_ffn=d_ffn,
        n_routed_experts=n_routed_experts,
        top_k=top_k,
        n_shared_experts=1
    )

    # 2. Speculative chunk of K=8 candidate tokens
    x_chunk = torch.randn(1, k_chunk, d_model)

    # 3. Compute Active Expert Union
    active_union, reuse_factor = moe.compute_chunk_expert_union(x_chunk)

    naive_expert_loads = k_chunk * top_k # 8 * 4 = 32 loads
    actual_unique_loads = len(active_union)

    print(f"Candidate Chunk Size (K)    : {k_chunk} tokens")
    print(f"Top-K per Token             : {top_k} experts")
    print(f"Naive Expert Loads          : {naive_expert_loads} loads")
    print(f"StreamSpec Unique Union     : {actual_unique_loads} unique experts ({active_union})")
    print(f"Expert Reuse / Savings Factor: {reuse_factor:.2f}x fewer expert transfers")

    # In an MoE with 32 experts, 8 tokens selecting 4 experts will overlap significantly
    assert actual_unique_loads < naive_expert_loads, "Expert unioning must produce fewer unique loads than naive"
    assert reuse_factor > 1.15, f"Expected reuse factor > 1.15x, got {reuse_factor:.2f}x"

    # 4. End-to-end forward pass execution
    out, telemetry = moe(x_chunk, active_expert_subset=active_union)
    assert out.shape == (1, k_chunk, d_model)
    assert telemetry["expert_sparsity_ratio"] > 0.50 # More than 50% of the MoE layer stayed unread

    print(f"Expert Sparsity Ratio (Unread Experts): {telemetry['expert_sparsity_ratio'] * 100:.1f}%")
    print("✅ DeepSeek Sparse MoE Expert Unioning verified.")

if __name__ == "__main__":
    test_sparse_moe_expert_unioning()
