# -*- coding: utf-8 -*-
"""
StreamSpec Benchmark SLA & Latency Regression Gate.
Verifies mathematical invariants, prefetch overlap, and I/O reduction bounds in CI.
"""

import os
import sys
import json
import time
from typing import Dict, Any, Optional
import torch

from streamspec.draft_worker import DraftWorker
from streamspec.target_streamer import TargetLayerStreamer
from streamspec.engine import StreamSpecEngine, StreamSpecConfig
from streamspec.telemetry.monitor import StreamSpecTelemetry

class StreamSpecSLAGate:
    """
    Automated Latency & SLA Regression Gate for StreamSpec.
    Verifies that speculative layer-streaming satisfies strict systems invariants.
    """

    def __init__(
        self,
        min_io_reduction: float = 1.20,
        min_prefetch_overlap: float = 0.80,
        min_acceptance_rate: float = 0.15,
        vocab_size: int = 150,
        prompt_len: int = 3,
        seed: int = 42
    ):
        self.min_io_reduction = min_io_reduction
        self.min_prefetch_overlap = min_prefetch_overlap
        self.min_acceptance_rate = min_acceptance_rate
        self.vocab_size = vocab_size
        self.prompt_len = prompt_len
        self.seed = seed

    def run(self, gen_tokens: int = 16, k_chunk: int = 4) -> Dict[str, Any]:
        torch.manual_seed(self.seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. Target Model Setup (with double-buffered prefetcher)
        target_model = TargetLayerStreamer(
            vocab_size=self.vocab_size,
            d_model=64,
            n_layers=2,
            n_q_heads=2,
            n_kv_heads=1,
            d_head=32,
            d_ffn=128,
            use_prefetcher=True
        )

        # 2. Distilled In-Memory Draft Speculator (aligned embedding & projection)
        draft_model = DraftWorker(
            vocab_size=self.vocab_size,
            d_model=64,
            n_layers=1,
            n_q_heads=2,
            n_kv_heads=1,
            d_head=32,
            d_ffn=128
        )
        draft_model.tok_embeddings.weight.data.copy_(target_model.tok_embeddings.weight.data)
        draft_model.lm_head.weight.data.copy_(target_model.lm_head.weight.data)

        prompt = torch.tensor([[5, 10, 15]])

        # 3. Vanilla Baseline Run (Prompt Prefill + Autoregressive Stream)
        vanilla_tokens = prompt.clone()
        start_vanilla = time.perf_counter()

        # Step 3a: Prefill prompt
        logits, vanilla_kv = target_model.stream_forward(prompt, kv_caches=None, pos_offset=0)
        next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        vanilla_tokens = torch.cat([vanilla_tokens, next_tok], dim=1)

        # Step 3b: Autoregressive decoding passes
        for _ in range(gen_tokens - 1):
            logits, vanilla_kv = target_model.stream_forward(
                vanilla_tokens[:, -1:], kv_caches=vanilla_kv, pos_offset=vanilla_tokens.shape[1] - 1
            )
            next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            vanilla_tokens = torch.cat([vanilla_tokens, next_tok], dim=1)
        time_vanilla = time.perf_counter() - start_vanilla

        # 4. StreamSpec Speculative Generation (Zero-Extra-Sweep + Adaptive K)
        config = StreamSpecConfig(
            k_chunk=k_chunk,
            temperature=0.0,
            max_new_tokens=gen_tokens,
            zero_extra_sweep=True,
            auto_tune_chunk=False
        )
        engine = StreamSpecEngine(draft_model, target_model, config)

        start_spec = time.perf_counter()
        spec_tokens, telemetry = engine.generate(prompt)
        time_spec = time.perf_counter() - start_spec

        # 5. Exact Mathematical Invariance Evaluation
        vanilla_generated = vanilla_tokens[0, self.prompt_len:self.prompt_len + gen_tokens]
        spec_generated = spec_tokens[0, self.prompt_len:self.prompt_len + gen_tokens]
        exact_match = bool(torch.equal(vanilla_generated, spec_generated))
        drift = 0.0 if exact_match else float((vanilla_generated != spec_generated).sum().item()) / gen_tokens

        # 6. Extract Telemetry Metrics
        io_reduction = telemetry.io_reduction_factor
        acceptance_rate = telemetry.acceptance_rate
        overlap_eff = target_model.prefetcher.overlap_efficiency if target_model.prefetcher is not None else 1.0

        # 7. Evaluate SLA Gates
        checks = {
            "exact_distribution_invariance": {
                "metric": "Exact Token Match",
                "actual": exact_match,
                "target": True,
                "passed": exact_match
            },
            "mathematical_drift": {
                "metric": "Output Distribution Drift",
                "actual": drift,
                "target": 0.0,
                "passed": drift == 0.0
            },
            "io_reduction_factor": {
                "metric": "Disk Sweep IO Reduction",
                "actual": round(io_reduction, 2),
                "target": f">= {self.min_io_reduction:.2f}x",
                "passed": io_reduction >= self.min_io_reduction
            },
            "prefetch_overlap_efficiency": {
                "metric": "Bus Prefetch Overlap Efficiency",
                "actual": f"{overlap_eff * 100:.1f}%",
                "target": f">= {self.min_prefetch_overlap * 100:.1f}%",
                "passed": overlap_eff >= self.min_prefetch_overlap
            },
            "token_acceptance_rate": {
                "metric": "Mean Token Acceptance Rate",
                "actual": f"{acceptance_rate * 100:.1f}%",
                "target": f">= {self.min_acceptance_rate * 100:.1f}%",
                "passed": acceptance_rate >= self.min_acceptance_rate
            }
        }

        all_passed = all(c["passed"] for c in checks.values())

        report = {
            "passed": all_passed,
            "device": device,
            "gen_tokens": gen_tokens,
            "k_chunk": k_chunk,
            "naive_sweeps": telemetry.naive_baseline_stream_passes,
            "spec_sweeps": telemetry.total_speculative_sweeps,
            "sync_sweeps_avoided": telemetry.total_sync_sweeps_avoided,
            "time_vanilla_s": round(time_vanilla, 4),
            "time_spec_s": round(time_spec, 4),
            "checks": checks
        }

        return report

    def render_markdown(self, report: Dict[str, Any]) -> str:
        status_badge = "PASSED" if report["passed"] else "FAILED"
        lines = [
            f"### StreamSpec CI SLA & Regression Gate [{status_badge}]",
            "",
            f"- **Execution Device:** `{report['device']}`",
            f"- **Target Tokens:** `{report['gen_tokens']}` (K* = {report['k_chunk']})",
            f"- **Naive Sweeps:** `{report['naive_sweeps']}` passes -> **StreamSpec Sweeps:** `{report['spec_sweeps']}` passes",
            f"- **Zero-Extra-Sweep Savings:** `{report['sync_sweeps_avoided']}` passes avoided",
            "",
            "| Invariant / SLA Check | Actual Metric | Target SLA | Status |",
            "| :--- | :--- | :--- | :--- |"
        ]

        for _, check in report["checks"].items():
            status = "PASSED" if check["passed"] else "FAILED"
            lines.append(f"| **{check['metric']}** | `{check['actual']}` | `{check['target']}` | `{status}` |")

        lines.append("")
        return "\n".join(lines)

def run_gate_cli(gen_tokens: int = 16, k_chunk: int = 4, output_json: Optional[str] = None) -> int:
    gate = StreamSpecSLAGate()
    report = gate.run(gen_tokens=gen_tokens, k_chunk=k_chunk)
    md = gate.render_markdown(report)
    print("\n" + md)

    step_summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if step_summary_path:
        try:
            with open(step_summary_path, "a", encoding="utf-8") as f:
                f.write("\n" + md + "\n")
        except Exception as e:
            print(f"[Warning] Could not write to GITHUB_STEP_SUMMARY: {e}", file=sys.stderr)

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Machine certificate saved to: {output_json}")

    return 0 if report["passed"] else 1

if __name__ == "__main__":
    sys.exit(run_gate_cli())
