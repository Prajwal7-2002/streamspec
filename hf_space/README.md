---
title: StreamSpec Studio
emoji: 🚀
colorFrom: yellow
colorTo: red
sdk: static
pinned: false
license: agpl-3.0
tags:
  - speculative-decoding
  - layer-streaming
  - deepseek-r1
  - deepseek-mla
  - inference-runtime
  - offloading
  - roofline-model
---

# StreamSpec Studio 🚀

### Asynchronous Speculative Layer-Streaming for Frontier & Futuristic Inference
Run 70B & Frontier Reasoning Models on Consumer Silicon at $10\times\text{--}15\times$ Streaming Speeds.

* **Interactive Web Application:** Built directly with client-side WebAssembly & JS (0ms cold-start).
* **GitHub Repository:** [Prajwal7-2002/streamspec](https://github.com/Prajwal7-2002/streamspec)
* **Architecture RFC:** [RFC-001 Systems Specification](https://github.com/Prajwal7-2002/streamspec/blob/main/docs/rfc-001-streamspec-speculative-layer-streaming.md)
* **PyPI Package:** [`pip install streamspec`](https://pypi.org/project/streamspec/)
* **Google Colab Cloud Runner:** [Open in Colab](https://colab.research.google.com/github/Prajwal7-2002/streamspec/blob/main/notebooks/streamspec_70b_benchmark.ipynb)
* **Author:** Prajwal Pujari ([@Prajwal7-2002](https://github.com/Prajwal7-2002))

---

### Systems Engineering Innovations:
1. **$K$-Chunked Speculative Streaming**: Decouples execution into an in-memory resident draft scout ($0.5\text{B}$) and single-pass batched target sweeps ($K$-chunk GEMM), restoring arithmetic intensity from $\mathcal{I} = 4.0 \to 64.0\text{ FLOPs/Byte}$.
2. **Double-Buffered Asynchronous Prefetcher (`DoubleBufferedPrefetcher`)**: Maintains alternating ping-pong buffers $\{B_0, B_1\}$ on device. Overlaps Layer $(\ell + 1)$ transfer with Layer $\ell$ compute, achieving **>90% bus overlap efficiency** without pipeline bubbles.
3. **Zero-Extra-Sweep Round Optimization**: Concatenates trailing accepted tokens directly into candidate chunks ($[x_{\text{trailing}} \;\Vert\; x_{\text{draft}}]$), cutting target disk sweeps by $2\times$ ($O(1)$ disk passes per round).
4. **Online Dynamic Adaptive $K$ Controller**: Exponential moving average filter modulates speculative chunk size $K \in [K_{\min}, K_{\max}]$ based on real-time empirical reasoning entropy.
5. **Exact Target Distribution Invariance**: Mathematical proof guaranteeing zero reasoning degradation or hallucination drift under Leviathan rejection sampling.

---

### Quickstart:
```bash
# 1. Install from PyPI
pip install streamspec

# 2. Run local verified benchmark simulator
streamspec demo

# 3. Or clone and execute on real Hugging Face model checkpoints
git clone https://github.com/Prajwal7-2002/streamspec.git
cd streamspec && python benchmarks/run_suite.py
```
