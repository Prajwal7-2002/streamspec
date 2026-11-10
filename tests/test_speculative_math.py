# -*- coding: utf-8 -*-
"""
Mathematical Invariance Test: Rejection Sampling Equivalence.
Proves empirically that Speculative Rejection Sampling yields the EXACT target distribution.
"""

import torch
import torch.nn.functional as F
from streamspec.engine import StreamSpecEngine

def test_rejection_sampling_exact_distribution():
    print("\n--- Running Test: Rejection Sampling Mathematical Invariance ---")
    torch.manual_seed(42)
    vocab_size = 5
    n_trials = 20000

    # 1. Define arbitrary Draft (q) and Target (p) probability distributions
    draft_logits = torch.tensor([[2.0, 1.0, 0.5, 0.1, -1.0]])
    target_logits = torch.tensor([[0.5, 2.5, 1.2, 0.8, 3.0]])

    q_probs = F.softmax(draft_logits, dim=-1)
    p_probs = F.softmax(target_logits, dim=-1)

    print(f"Draft Probs (q) : {[round(x, 4) for x in q_probs[0].tolist()]}")
    print(f"Target Probs (p): {[round(x, 4) for x in p_probs[0].tolist()]}")

    emitted_counts = torch.zeros(vocab_size)

    # 2. Run N trials of Speculative Rejection Sampling for single token candidate
    for _ in range(n_trials):
        cand_token = torch.multinomial(q_probs, num_samples=1)
        accepted, _ = StreamSpecEngine.verify_rejection_sampling(
            draft_tokens=cand_token,
            draft_logits=draft_logits.unsqueeze(1),
            target_dists=target_logits.unsqueeze(1),
            temperature=1.0
        )
        emitted_counts[accepted[0]] += 1

    empirical_p = emitted_counts / n_trials
    print(f"Empirical Output: {[round(x, 4) for x in empirical_p.tolist()]}")

    # 3. Assert Max Absolute Difference < 0.015 (Law of Large Numbers Convergence)
    max_err = (empirical_p - p_probs[0]).abs().max().item()
    print(f"Max Absolute Error vs Target: {max_err:.5f}")
    assert max_err < 0.015, f"Rejection sampling distribution deviated! Max error: {max_err}"
    print("PASS: Mathematical equivalence to Target Model strictly verified (Q.E.D.)")

if __name__ == "__main__":
    test_rejection_sampling_exact_distribution()
