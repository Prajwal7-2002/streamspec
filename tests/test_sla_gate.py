# -*- coding: utf-8 -*-
"""
Unit Test: StreamSpec Benchmark SLA & Latency Regression Gate.
Verifies that the automated SLA gate correctly evaluates mathematical invariance,
I/O reduction bounds, and generates valid CI reports.
"""

import os
import pytest
import torch
from streamspec.benchmarks.sla_gate import StreamSpecSLAGate, run_gate_cli

def test_sla_gate_pass():
    gate = StreamSpecSLAGate(
        min_io_reduction=1.20,
        min_acceptance_rate=0.15,
        vocab_size=300,
        prompt_len=4,
        seed=42
    )
    report = gate.run(gen_tokens=16, k_chunk=4)

    assert report["passed"] is True, f"Expected SLA gate to pass, but failed: {report}"
    assert report["checks"]["exact_distribution_invariance"]["passed"] is True
    assert report["checks"]["mathematical_drift"]["actual"] == 0.0
    assert report["checks"]["io_reduction_factor"]["passed"] is True
    assert report["checks"]["prefetch_overlap_efficiency"]["passed"] is True
    assert report["checks"]["token_acceptance_rate"]["passed"] is True

def test_sla_gate_hardware_aware_threshold():
    # Verify device-aware threshold: 0.80 on CUDA DMA, 0.50 on CPU software emulation
    gate_auto = StreamSpecSLAGate()
    expected_default = 0.80 if torch.cuda.is_available() else 0.50
    assert gate_auto.min_prefetch_overlap == expected_default

    # Verify explicit override is preserved
    gate_custom = StreamSpecSLAGate(min_prefetch_overlap=0.75)
    assert gate_custom.min_prefetch_overlap == 0.75

def test_sla_gate_violation_detection():
    # Set an impossible I/O reduction bound to trigger intentional SLA violation
    gate = StreamSpecSLAGate(
        min_io_reduction=50.0,
        vocab_size=300,
        prompt_len=4,
        seed=42
    )
    report = gate.run(gen_tokens=12, k_chunk=4)

    assert report["passed"] is False
    assert report["checks"]["io_reduction_factor"]["passed"] is False

def test_sla_gate_markdown_rendering(tmp_path):
    gate = StreamSpecSLAGate(vocab_size=200, prompt_len=4, seed=42)
    report = gate.run(gen_tokens=12, k_chunk=4)
    md = gate.render_markdown(report)

    assert "### StreamSpec CI SLA & Regression Gate" in md
    assert "Exact Token Match" in md
    assert "Disk Sweep IO Reduction" in md

    # Test output to JSON
    json_path = tmp_path / "cert.json"
    exit_code = run_gate_cli(gen_tokens=12, k_chunk=4, output_json=str(json_path))
    assert exit_code == 0
    assert json_path.exists()
