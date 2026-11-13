# -*- coding: utf-8 -*-
"""
Unit Test: Adaptive Speculative Chunk Controller (Online PID / EWMA).
Verifies dynamic expansion during high alignment and contraction during low alignment.
"""

import torch
from streamspec.engine import AdaptiveChunkController, StreamSpecEngine, StreamSpecConfig
from streamspec.target_streamer import TargetLayerStreamer
from tests.test_streaming_pipeline import AlignedMockDraftWorker

def test_adaptive_chunk_controller_unit():
    ctrl = AdaptiveChunkController(
        initial_k=4,
        min_k=2,
        max_k=10,
        ema_alpha=0.4,
        high_threshold=0.75,
        low_threshold=0.40
    )

    assert ctrl.k == 4

    # High acceptance scenario (gamma = proposed)
    for _ in range(6):
        ctrl.update(accepted_count=4, proposed_count=4)
    assert ctrl.k > 4, "High acceptance must grow speculative chunk size K"
    assert ctrl.k <= 10, "K must not exceed max_k"

    # Low acceptance scenario (gamma = 1, proposed = K)
    for _ in range(10):
        ctrl.update(accepted_count=1, proposed_count=ctrl.k)
    assert ctrl.k < 4, "Low acceptance must shrink speculative chunk size K"
    assert ctrl.k >= 2, "K must not drop below min_k"

def test_adaptive_chunk_in_engine():
    torch.manual_seed(42)
    vocab_size = 200
    prompt_len = 3
    max_new = 16

    target_model = TargetLayerStreamer(
        vocab_size=vocab_size,
        d_model=64,
        n_layers=2,
        n_q_heads=2,
        n_kv_heads=1,
        d_head=32,
        d_ffn=128,
        use_prefetcher=True
    )

    draft_model = AlignedMockDraftWorker(
        target_streamer=target_model,
        agreement_rate=0.85,
        vocab_size=vocab_size,
        d_model=32,
        n_layers=1,
        n_q_heads=2,
        n_kv_heads=1,
        d_head=16,
        d_ffn=64
    )

    config = StreamSpecConfig(
        k_chunk=3,
        min_k=2,
        max_k=8,
        auto_tune_chunk=True,
        zero_extra_sweep=True,
        max_new_tokens=max_new
    )

    engine = StreamSpecEngine(draft_model, target_model, config)
    prompt = torch.randint(0, vocab_size, (1, prompt_len))
    out, tel = engine.generate(prompt)

    assert len(tel.adaptive_k_history) > 0, "Adaptive chunk history must be recorded in telemetry"
    assert out.shape[1] - prompt_len >= max_new
    print("Adaptive Chunk Controller in Engine Verified Successfully!")

if __name__ == "__main__":
    test_adaptive_chunk_controller_unit()
    test_adaptive_chunk_in_engine()

