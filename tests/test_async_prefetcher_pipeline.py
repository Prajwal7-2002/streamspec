# -*- coding: utf-8 -*-
"""
Unit Test: Asynchronous Double-Buffered Prefetcher Integration in TargetLayerStreamer.
Verifies numerical equivalence between synchronous streaming and double-buffered asynchronous prefetching.
"""

import torch
from streamspec.target_streamer import TargetLayerStreamer

def test_target_streamer_prefetcher_equivalence():
    torch.manual_seed(123)
    vocab_size = 300
    d_model = 64
    n_layers = 4
    seq_len = 5

    # 1. Create target streamer with prefetcher enabled
    streamer_prefetch = TargetLayerStreamer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_q_heads=4,
        n_kv_heads=2,
        d_head=16,
        d_ffn=176,
        use_prefetcher=True
    )

    # 2. Create target streamer with prefetcher disabled (synchronous fallback)
    streamer_sync = TargetLayerStreamer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_q_heads=4,
        n_kv_heads=2,
        d_head=16,
        d_ffn=176,
        use_prefetcher=False
    )

    # Copy identical static weights and layer weights
    streamer_sync.tok_embeddings.load_state_dict(streamer_prefetch.tok_embeddings.state_dict())
    streamer_sync.final_norm.load_state_dict(streamer_prefetch.final_norm.state_dict())
    streamer_sync.lm_head.load_state_dict(streamer_prefetch.lm_head.state_dict())
    streamer_sync.layer_weight_bank = [
        {k: v.clone() for k, v in layer.items()}
        for layer in streamer_prefetch.layer_weight_bank
    ]

    input_ids = torch.randint(0, vocab_size, (1, seq_len))

    # Run prefetcher pass
    logits_prefetch, kv_prefetch = streamer_prefetch.stream_forward(input_ids)

    # Run sync pass
    logits_sync, kv_sync = streamer_sync.stream_forward(input_ids)

    # Numerically identical verification
    assert torch.allclose(logits_prefetch, logits_sync, atol=1e-5), (
        "Logits between double-buffered prefetcher and sync streamer must match!"
    )

    for l in range(n_layers):
        k_p, v_p = kv_prefetch[l]
        k_s, v_s = kv_sync[l]
        assert torch.allclose(k_p, k_s, atol=1e-5), f"K cache for layer {l} must match"
        assert torch.allclose(v_p, v_s, atol=1e-5), f"V cache for layer {l} must match"

    assert streamer_prefetch.prefetcher.layers_prefetched > 0, "Prefetcher must report prefetched layers"
    assert streamer_prefetch.prefetcher.overlap_efficiency >= 0.0, "Overlap efficiency must be non-negative"

    print(f"Double-buffered prefetcher verified: overlap efficiency = {streamer_prefetch.prefetcher.overlap_efficiency * 100:.1f}%")

if __name__ == "__main__":
    test_target_streamer_prefetcher_equivalence()

