# -*- coding: utf-8 -*-
"""
StreamSpec Engine: Universal Asynchronous Speculative Layer-Streaming Orchestrator.
Combines in-memory draft speculation with single-pass memory-mapped layer verification.
"""

from dataclasses import dataclass
import time
import torch
import torch.nn.functional as F

from .draft_worker import DraftWorker
from .target_streamer import TargetLayerStreamer
from .telemetry.monitor import StreamSpecTelemetry

@dataclass
class StreamSpecConfig:
    k_chunk: int = 6                # Initial speculative chunk size
    temperature: float = 0.0        # 0.0 = greedy argmax, >0.0 = stochastic sampling
    max_new_tokens: int = 32        # Generation ceiling
    auto_tune_chunk: bool = True    # Dynamic adaptive chunk size controller
    min_k: int = 2                  # Lower bound on adaptive K
    max_k: int = 16                 # Upper bound on adaptive K
    zero_extra_sweep: bool = True   # Enable zero-extra-sweep concatenation optimization

class AdaptiveChunkController:
    """
    Online PID/EWMA controller that dynamically tunes speculative chunk size K
    based on the rolling empirical acceptance rate alpha = gamma / K.
    
    If acceptance is high (e.g. >= 0.75), growing K yields more accepted tokens per sweep.
    If acceptance is low (e.g. <= 0.40), shrinking K avoids wasting drafting compute on rejected tokens.
    """
    def __init__(
        self,
        initial_k: int = 6,
        min_k: int = 2,
        max_k: int = 16,
        ema_alpha: float = 0.35,
        high_threshold: float = 0.75,
        low_threshold: float = 0.40,
    ):
        self.k = initial_k
        self.min_k = min_k
        self.max_k = max_k
        self.ema_alpha = ema_alpha
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.rolling_acceptance: float = 0.70  # initial prior
        self.history = []

    def update(self, accepted_count: int, proposed_count: int) -> int:
        if proposed_count <= 0:
            return self.k
            
        rate = accepted_count / proposed_count
        self.rolling_acceptance = (
            (1.0 - self.ema_alpha) * self.rolling_acceptance + self.ema_alpha * rate
        )
        self.history.append((accepted_count, proposed_count, self.rolling_acceptance, self.k))

        if self.rolling_acceptance >= self.high_threshold and self.k < self.max_k:
            # High alignment: expand chunk size
            step = 2 if self.rolling_acceptance >= 0.85 else 1
            self.k = min(self.max_k, self.k + step)
        elif self.rolling_acceptance <= self.low_threshold and self.k > self.min_k:
            # Low alignment: contract chunk size
            step = 2 if self.rolling_acceptance <= 0.25 else 1
            self.k = max(self.min_k, self.k - step)

        return self.k

class StreamSpecEngine:
    def __init__(
        self,
        draft_model: DraftWorker,
        target_model: TargetLayerStreamer,
        config: StreamSpecConfig = None
    ):
        self.draft_model = draft_model
        self.target_model = target_model
        self.config = config or StreamSpecConfig()
        self.telemetry = StreamSpecTelemetry()

    @staticmethod
    def sample_token(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
        """Sample from logits using greedy argmax or temperature scaled distribution."""
        if temperature <= 1e-5:
            return torch.argmax(logits, dim=-1, keepdim=True)
        probs = F.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @staticmethod
    def verify_rejection_sampling(
        draft_tokens: torch.Tensor,
        draft_logits: torch.Tensor,
        target_dists: torch.Tensor,
        temperature: float = 0.0
    ) -> tuple[list[int], int]:
        """
        Exact Mathematical Rejection Sampling (Leviathan et al. 2023).
        Guarantees 100% mathematical distribution equivalence to sampling directly from target model.
        Returns:
          accepted_tokens: list of token IDs
          gamma: count of accepted tokens
        """
        k = draft_tokens.shape[1]
        accepted = []

        if temperature <= 1e-5:
            # Deterministic greedy verification
            target_preds = torch.argmax(target_dists, dim=-1) # [1, K]
            for i in range(k):
                cand = draft_tokens[0, i].item()
                target_choice = target_preds[0, i].item()
                if cand == target_choice:
                    accepted.append(cand)
                else:
                    # First divergence: reject cand, emit true target choice, stop chunk
                    accepted.append(target_choice)
                    return accepted, len(accepted)
            
            return accepted, len(accepted)

        # Stochastic Rejection Sampling
        q = F.softmax(draft_logits / temperature, dim=-1)  # [1, K, Vocab]
        p = F.softmax(target_dists / temperature, dim=-1) # [1, K, Vocab]

        for i in range(k):
            token_id = draft_tokens[0, i].item()
            q_prob = q[0, i, token_id].item()
            p_prob = p[0, i, token_id].item()

            alpha = min(1.0, p_prob / max(q_prob, 1e-12))
            u = torch.rand(1).item()

            if u < alpha:
                accepted.append(token_id)
            else:
                # Rejection: sample from normalized residual distribution
                residual = torch.clamp(p[0, i] - q[0, i], min=0.0)
                norm = residual.sum()
                if norm > 0:
                    residual_probs = residual / norm
                    resampled = torch.multinomial(residual_probs, 1).item()
                else:
                    resampled = torch.argmax(p[0, i]).item()
                accepted.append(resampled)
                return accepted, len(accepted)

        return accepted, len(accepted)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, StreamSpecTelemetry]:
        """
        Executes the Speculative Layer-Streaming Generation Loop.
        Input: input_ids [1, PromptLen]
        Output: generated_ids [1, PromptLen + GeneratedLen]
        """
        self.telemetry.start()
        curr_ids = input_ids.clone()
        max_new = self.config.max_new_tokens

        # Online adaptive chunk controller
        if self.config.auto_tune_chunk:
            controller = AdaptiveChunkController(
                initial_k=self.config.k_chunk,
                min_k=self.config.min_k,
                max_k=self.config.max_k,
            )
        else:
            controller = None

        # Initial Prefill Pass on Target Model (single pass for prompt)
        target_kv = None
        target_logits, target_kv = self.target_model.stream_forward(curr_ids, kv_caches=target_kv, pos_offset=0)
        last_target_logits = target_logits[:, -1:, :] # Distribution predicting the first new token

        # Initialize Draft KV cache
        draft_kv = None
        _, draft_kv = self.draft_model(curr_ids, kv_caches=draft_kv)

        tokens_generated = 0
        trailing_token = None

        while tokens_generated < max_new:
            # 1. Determine speculative chunk size K
            if controller is not None:
                chunk_size = min(controller.k, max_new - tokens_generated)
            else:
                chunk_size = min(self.config.k_chunk, max_new - tokens_generated)
            chunk_size = max(1, chunk_size)

            # 2. Draft Phase: fast in-memory draft generation of K tokens
            draft_tokens, draft_logits, _ = self.draft_model.generate_draft_chunk(
                curr_ids, k_chunk=chunk_size, kv_caches=draft_kv
            )

            # 3. Target Streaming Phase (Zero-Extra-Sweep Optimization)
            if self.config.zero_extra_sweep:
                if trailing_token is None:
                    # Initial round: target streams draft_tokens directly
                    target_chunk_logits, target_kv = self.target_model.stream_forward(
                        draft_tokens, kv_caches=target_kv, pos_offset=curr_ids.shape[1]
                    )
                    target_dists = torch.cat([last_target_logits, target_chunk_logits[:, :-1, :]], dim=1)
                    bonus_dist = target_chunk_logits[:, -1:, :]
                    curr_kv_base = curr_ids.shape[1]
                else:
                    # Subsequent rounds: combine trailing accepted token with draft candidates into ONE single pass!
                    sweep_input = torch.cat([trailing_token, draft_tokens], dim=1)
                    target_chunk_logits, target_kv = self.target_model.stream_forward(
                        sweep_input, kv_caches=target_kv, pos_offset=curr_ids.shape[1] - 1
                    )
                    target_dists = target_chunk_logits[:, :-1, :]
                    bonus_dist = target_chunk_logits[:, -1:, :]
                    curr_kv_base = curr_ids.shape[1] - 1
                    self.telemetry.record_sync_sweep_avoided()

                # 4. Exact Verification
                accepted_chunk, gamma = self.verify_rejection_sampling(
                    draft_tokens,
                    draft_logits,
                    target_dists,
                    temperature=self.config.temperature
                )

                # 5. Handle trailing token & KV cache trimming
                if gamma == chunk_size:
                    # All draft tokens accepted!
                    if tokens_generated + gamma < max_new:
                        bonus_token = self.sample_token(bonus_dist[:, 0, :], self.config.temperature).item()
                        accepted_chunk.append(bonus_token)
                        gamma += 1
                        trailing_token = torch.tensor([[bonus_token]], device=curr_ids.device)
                        valid_kv_len = curr_kv_base + 1 + chunk_size
                    else:
                        trailing_token = None
                        valid_kv_len = curr_kv_base + chunk_size
                else:
                    # Rejection occurred at index (gamma - 1)
                    resampled_token = accepted_chunk[-1]
                    trailing_token = torch.tensor([[resampled_token]], device=curr_ids.device)
                    rejected_idx = gamma - 1
                    valid_kv_len = curr_kv_base + 1 + rejected_idx

                # Trim target KV cache to valid non-speculative prefix
                if target_kv is not None:
                    target_kv = [
                        (k[:, :valid_kv_len, :, :], v[:, :valid_kv_len, :, :])
                        for (k, v) in target_kv
                    ]
            else:
                # Legacy path with separate sync sweep
                target_chunk_logits, target_kv = self.target_model.stream_forward(
                    draft_tokens, kv_caches=target_kv, pos_offset=curr_ids.shape[1]
                )
                target_dists = torch.cat([last_target_logits, target_chunk_logits[:, :-1, :]], dim=1)
                bonus_dist = target_chunk_logits[:, -1:, :]
                accepted_chunk, gamma = self.verify_rejection_sampling(
                    draft_tokens, draft_logits, target_dists, temperature=self.config.temperature
                )
                if gamma == chunk_size and tokens_generated + gamma < max_new:
                    bonus_token = self.sample_token(bonus_dist[:, 0, :], self.config.temperature).item()
                    accepted_chunk.append(bonus_token)
                    gamma += 1

            # Append accepted tokens to active sequence
            accepted_tensor = torch.tensor([accepted_chunk], device=curr_ids.device)
            curr_ids = torch.cat([curr_ids, accepted_tensor], dim=1)
            tokens_generated += gamma

            # Record telemetry
            bytes_streamed = sum(
                sum(t.numel() * t.element_size() for t in b.values())
                for b in self.target_model.layer_weight_bank
            )
            rolling_a = controller.rolling_acceptance if controller else None
            self.telemetry.record_chunk(chunk_size, gamma, bytes_streamed, current_k=chunk_size, rolling_alpha=rolling_a)

            # Update adaptive controller
            if controller is not None:
                controller.update(gamma, chunk_size)

            # If legacy path, sync target model with extra sweep
            if not self.config.zero_extra_sweep:
                last_target_logits, target_kv = self.target_model.stream_forward(
                    curr_ids[:, -1:], kv_caches=target_kv, pos_offset=curr_ids.shape[1] - 1
                )

            # Synchronize draft model
            _, draft_kv = self.draft_model(curr_ids, kv_caches=None)

        return curr_ids, self.telemetry
