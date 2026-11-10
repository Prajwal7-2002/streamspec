# -*- coding: utf-8 -*-
"""
Systems Director Live Telemetry & Roofline Monitor for StreamSpec.
Tracks physical memory bandwidth, draft acceptance rate, and I/O speedup.
"""

import time

class StreamSpecTelemetry:
    def __init__(self):
        self.start_time = None
        self.total_tokens_generated = 0
        self.total_speculative_sweeps = 0
        self.total_draft_tokens_proposed = 0
        self.total_draft_tokens_accepted = 0
        self.total_bytes_streamed = 0
        self.naive_baseline_stream_passes = 0

    def start(self):
        self.start_time = time.perf_counter()

    def record_chunk(self, proposed_k: int, accepted_gamma: int, bytes_streamed: int):
        self.total_speculative_sweeps += 1
        self.total_draft_tokens_proposed += proposed_k
        self.total_draft_tokens_accepted += accepted_gamma
        self.total_tokens_generated += accepted_gamma
        self.total_bytes_streamed += bytes_streamed
        # In naive single-token streaming, every token requires a separate disk sweep
        self.naive_baseline_stream_passes += accepted_gamma

    @property
    def acceptance_rate(self) -> float:
        if self.total_draft_tokens_proposed == 0:
            return 0.0
        return self.total_draft_tokens_accepted / self.total_draft_tokens_proposed

    @property
    def io_reduction_factor(self) -> float:
        if self.total_speculative_sweeps == 0:
            return 1.0
        return self.naive_baseline_stream_passes / self.total_speculative_sweeps

    @property
    def tokens_per_second(self) -> float:
        if self.start_time is None:
            return 0.0
        elapsed = time.perf_counter() - self.start_time
        if elapsed <= 0:
            return 0.0
        return self.total_tokens_generated / elapsed

    def print_director_dashboard(self):
        elapsed = time.perf_counter() - self.start_time if self.start_time else 0.0
        print("\n" + "=" * 68)
        print("  STREAMSPEC SYSTEMS DIRECTOR TELEMETRY DASHBOARD")
        print("=" * 68)
        print(f"  Total Tokens Generated      : {self.total_tokens_generated}")
        print(f"  Wall-Clock Elapsed Time     : {elapsed:.2f} s")
        print(f"  Effective Throughput        : {self.tokens_per_second:.2f} tokens/sec")
        print(f"  Mean Acceptance Rate (alpha): {self.acceptance_rate * 100:.1f}%")
        print(f"  Speculative Stream Sweeps   : {self.total_speculative_sweeps}")
        print(f"  Naive Streaming Sweeps Avoid: {self.naive_baseline_stream_passes}")
        print(f"  Hardware I/O Bandwidth Saved: {self.io_reduction_factor:.2f}x reduction")
        print("=" * 68 + "\n")

