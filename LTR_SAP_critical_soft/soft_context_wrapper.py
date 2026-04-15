"""
SoftContextWrapper: wraps a SEDD model to inject soft embeddings at specified positions.

The wrapper replicates the exact forward pass of SEDD (model/transformer.py SEDD.forward)
but replaces discrete token embeddings at designated positions with pre-computed soft
embeddings (probability-weighted averages of the embedding table).

Usage:
    wrapper = SoftContextWrapper(model)
    wrapper.set_soft_context(positions=[3, 4], embeddings=soft_emb_tensor)
    score_fn = get_score_fn(wrapper, sampling=True)
    # ... use score_fn in denoising loop ...
    wrapper.clear_soft_context()
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftContextWrapper(nn.Module):
    """Wraps SEDD to inject soft embeddings while preserving the full forward pass."""

    def __init__(self, model):
        super().__init__()
        self.model = model

        self._soft_positions = []
        self._soft_embeddings = None

    # -- delegate attribute access to the wrapped model so that external code
    #    (e.g. get_model_fn, get_score_fn, config checks) works transparently --

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    # ----- public API -----

    def set_soft_context(self, positions, embeddings):
        """Register soft embeddings for specific sequence positions.

        Args:
            positions: list[int] — sequence positions to override.
            embeddings: Tensor [n_positions, hidden_dim] — soft embeddings for each.
        """
        self._soft_positions = list(positions)
        self._soft_embeddings = embeddings

    def clear_soft_context(self):
        """Remove all soft context (revert to normal discrete embedding)."""
        self._soft_positions = []
        self._soft_embeddings = None

    # ----- forward pass (mirrors SEDD.forward exactly) -----

    def forward(self, indices, sigma):
        """Forward pass identical to SEDD.forward except for soft-embedding injection.

        Args:
            indices: [B, L] integer token IDs (ground-truth at prefix, MASK elsewhere).
            sigma:   [B] noise level.

        Returns:
            [B, L, V] score tensor (same as SEDD.forward).
        """
        x = self.model.vocab_embed(indices)

        if self._soft_positions and self._soft_embeddings is not None:
            for i, pos in enumerate(self._soft_positions):
                x[:, pos] = self._soft_embeddings[i].to(x.dtype)

        c = F.silu(self.model.sigma_map(sigma))
        rotary_cos_sin = self.model.rotary_emb(x)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for block in self.model.blocks:
                x = block(x, rotary_cos_sin, c, seqlens=None)
            x = self.model.output_layer(x, c)

        if self.model.scale_by_sigma:
            esigm1_log = (
                torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1)
                .log()
                .to(x.dtype)[:, None, None]
            )
            x = x - esigm1_log - np.log(x.shape[-1] - 1)

        x = torch.scatter(x, -1, indices[..., None], torch.zeros_like(x[..., :1]))
        return x


def build_soft_embedding(p_soft, embedding_table, vocab_size=None):
    """Compute a soft embedding from a probability distribution over the vocabulary.

    Args:
        p_soft: [V] or [V'] probability vector (may include MASK dimension).
        embedding_table: nn.Parameter [total_vocab, hidden_dim] (includes MASK row).
        vocab_size: int, number of real tokens (excluding MASK). If None, uses
                    embedding_table.shape[0].

    Returns:
        [hidden_dim] soft embedding tensor.
    """
    if vocab_size is None:
        vocab_size = embedding_table.shape[0]

    p = p_soft[:vocab_size]
    p = p / p.sum().clamp(min=1e-30)

    return (p[:, None] * embedding_table[:vocab_size]).sum(dim=0)
