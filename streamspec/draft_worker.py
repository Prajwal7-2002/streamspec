# -*- coding: utf-8 -*-
"""
Fast In-Memory Draft Generation Worker.
Generates K candidate tokens at high operational frequency before verification sweep.
"""

import torch
import torch.nn as nn
from .architectures.llama import ModernTransformerBlock, RMSNorm

class DraftWorker(nn.Module):
    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 256,       # Lightweight draft model (e.g. 0.1B - 0.5B)
        n_layers: int = 4,
        n_q_heads: int = 4,
        n_kv_heads: int = 2,
        d_head: int = 64,
        d_ffn: int = 704,
        device: str = "cpu"
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.device = device

        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            ModernTransformerBlock(d_model, n_q_heads, n_kv_heads, d_head, d_ffn)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.to(device)

    def forward(self, input_ids: torch.Tensor, kv_caches: list = None, pos_offset: int = 0):
        h = self.tok_embeddings(input_ids)
        new_kvs = []
        for i, layer in enumerate(self.layers):
            layer_kv = kv_caches[i] if kv_caches else None
            h, new_kv = layer(h, layer_kv, pos_offset=pos_offset)
            new_kvs.append(new_kv)
        normed = self.final_norm(h)
        logits = self.lm_head(normed)
        return logits, new_kvs

    @torch.no_grad()
    def generate_draft_chunk(
        self,
        current_input_ids: torch.Tensor,
        k_chunk: int = 8,
        kv_caches: list = None
    ) -> tuple[torch.Tensor, torch.Tensor, list]:
        """
        Autoregressively generate K candidate tokens.
        Returns:
          draft_tokens: [1, K]
          draft_logits: [1, K, Vocab]
          draft_kvs
        """
        draft_tokens = []
        draft_logits_list = []
        curr_token = current_input_ids[:, -1:]
        pos_offset = current_input_ids.shape[1] - 1

        for step in range(k_chunk):
            logits, kv_caches = self.forward(curr_token, kv_caches, pos_offset=pos_offset + step)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            draft_tokens.append(next_token)
            draft_logits_list.append(next_token_logits.unsqueeze(1))
            curr_token = next_token

        draft_tokens_tensor = torch.cat(draft_tokens, dim=1) # [1, K]
        draft_logits_tensor = torch.cat(draft_logits_list, dim=1) # [1, K, Vocab]
        return draft_tokens_tensor, draft_logits_tensor, kv_caches

