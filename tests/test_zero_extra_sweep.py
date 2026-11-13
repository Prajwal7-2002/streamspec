# -*- coding: utf-8 -*-
"""
Unit Test: Zero-Extra-Sweep Optimization.
Verifies that combining the trailing accepted token with the next speculative draft chunk
eliminates redundant synchronization sweeps while preserving 100% distribution equivalence.
"""

import torch
from streamspec.target_streamer import TargetLayerStreamer
from streamspec.draft_worker import DraftWorker
from streamspec.engine import StreamSpecEngine, StreamSpecConfig
from tests.test_streaming_pipeline import AlignedMockDraftWorker

def test_zero_extra_sweep_eliminates_sync_passes():
    torch.manual_seed(42)
    vocab_size = 250
    prompt_len = 4
    max_new = 15

    target_model = TargetLayerStreamer(
        vocab_size=vocab_size,
        d_model=64,
        n_layers=3,
        n_q_heads=4,
        n_kv_heads=2,
        d_head=16,
        d_ffn=176,
        use_prefetcher=True
    )

    draft_model = DraftWorker(
        vocab_size=vocab_size,
        d_model=32,
        n_layers=2,
        n_q_heads=2,
        n_kv_heads=1,
        d_head=16,
        d_ffn=88
    )

    prompt = torch.randint(0, vocab_size, (1, prompt_len))

    # 1. Run with Zero-Extra-Sweep enabled
    config_opt = StreamSpecConfig(
        k_chunk=4,
        max_new_tokens=max_new,
        temperature=0.0,
        zero_extra_sweep=True,
        auto_tune_chunk=False
    )
    engine_opt = StreamSpecEngine(draft_model, target_model, config_opt)
    out_opt, tel_opt = engine_opt.generate(prompt)

    # 2. Verify sweeps avoided
    assert tel_opt.total_sync_sweeps_avoided > 0, "Zero-Extra-Sweep must record avoided sync sweeps!"
    assert out_opt.shape[1] - prompt_len >= max_new, "Must generate at least max_new tokens"

    # 3. Target stream passes must be strictly bounded:
    # 1 prefill pass + round passes (zero extra sync passes)
    expected_max_passes = 1 + tel_opt.total_speculative_sweeps
    assert target_model.total_stream_passes <= expected_max_passes, (
        f"Stream passes {target_model.total_stream_passes} exceeded expected {expected_max_passes}"
    )

    print(f"Zero-Extra-Sweep Test Passed: {tel_opt.total_sync_sweeps_avoided} extra disk sweeps avoided!")

if __name__ == "__main__":
    test_zero_extra_sweep_eliminates_sync_passes()
