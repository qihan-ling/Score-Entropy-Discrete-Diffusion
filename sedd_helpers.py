"""
Shared helpers for SEDD exploration and experiments.

Reusable across any sampling variant (strict LTR, windowed, standard parallel).
Does not depend on any specific sampling strategy.
"""

import torch
import numpy as np
from transformers import GPT2TokenizerFast
from load_model import load_model
from model import utils as mutils


def load_sedd_model(model_path, device):
    """Load SEDD model, graph, noise, and tokenizer."""
    model, graph, noise = load_model(model_path, device)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    return model, graph, noise, tokenizer


def get_sampling_score_fn(model):
    """Get the score function used for sampling (returns exp(log_score))."""
    return mutils.get_score_fn(model, train=False, sampling=True)


def get_log_score_fn(model):
    """Get the raw log-score function (used for training / inspection)."""
    return mutils.get_score_fn(model, train=False, sampling=False)


# ---------------------------------------------------------------------------
# Causal mode toggle
# ---------------------------------------------------------------------------

def set_causal_mode(model, causal):
    """
    Monkey-patch all DDiTBlock instances to use causal or non-causal attention.

    Works by replacing the forward method of each block with a wrapper that
    overrides the attention call's causal flag.
    """
    from model.transformer import standard_attention_varlen, DDiTBlock
    from model.fused_add_dropout_scale import modulate_fused
    from einops import rearrange
    from model import rotary as rotary_module

    for block in model.blocks:
        if not isinstance(block, DDiTBlock):
            continue

        original_n_heads = block.n_heads
        original_dropout = block.dropout

        def make_forward(blk, _causal):
            def patched_forward(x, rotary_cos_sin, c, seqlens=None):
                batch_size, seq_len = x.shape[0], x.shape[1]
                bias_dropout_scale_fn = blk._get_bias_dropout_scale()

                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    blk.adaLN_modulation(c)[:, None].chunk(6, dim=2)
                )

                x_skip = x
                x = modulate_fused(blk.norm1(x), shift_msa, scale_msa)

                qkv = blk.attn_qkv(x)
                qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=blk.n_heads)
                with torch.cuda.amp.autocast(enabled=False):
                    cos, sin = rotary_cos_sin
                    qkv = rotary_module.apply_rotary_pos_emb(
                        qkv, cos.to(qkv.dtype), sin.to(qkv.dtype)
                    )
                qkv = rearrange(qkv, "b s ... -> (b s) ...")
                if seqlens is None:
                    cu_seqlens = torch.arange(
                        0, (batch_size + 1) * seq_len, step=seq_len,
                        dtype=torch.int32, device=qkv.device,
                    )
                else:
                    cu_seqlens = seqlens.cumsum(-1)

                x = standard_attention_varlen(
                    qkv, cu_seqlens, seq_len, 0., causal=_causal
                )
                x = rearrange(x, "(b s) h d -> b s (h d)", b=batch_size)

                x = bias_dropout_scale_fn(
                    blk.attn_out(x), None, gate_msa, x_skip, blk.dropout
                )
                x = bias_dropout_scale_fn(
                    blk.mlp(modulate_fused(blk.norm2(x), shift_mlp, scale_mlp)),
                    None, gate_mlp, x, blk.dropout,
                )
                return x

            return patched_forward

        block.forward = make_forward(block, causal)


# ---------------------------------------------------------------------------
# Re-noising (distance-based forgetting)
# ---------------------------------------------------------------------------

def renoise_fn(x, graph, frontier_pos, prefix_len, renoise_sigma):
    """
    Distance-based forgetting for strict LTR generation.

    For committed positions between prefix_len and frontier_pos-1:
      - distance 1-4 from frontier: probabilistic re-masking with increasing sigma
      - distance >= 5: deterministic re-masking (forgotten)
    Prefix positions are never touched.

    Args:
        x: [B, L] token ids
        graph: Absorbing graph instance
        frontier_pos: current frontier position
        prefix_len: number of fixed prefix tokens
        renoise_sigma: base sigma controlling noise intensity
    Returns:
        x: modified in-place and returned
    """
    MASK = graph.dim - 1

    for dist in range(1, frontier_pos - prefix_len + 1):
        j = frontier_pos - dist
        if j < prefix_len:
            break
        if dist >= 5:
            x[:, j] = MASK
        else:
            sigma_j = renoise_sigma * (dist / 5.0)
            move_chance = 1 - torch.tensor(-sigma_j, device=x.device, dtype=torch.float32).exp()
            move = torch.rand(x.shape[0], device=x.device) < move_chance
            x[:, j] = torch.where(move, torch.tensor(MASK, device=x.device), x[:, j])
    return x


# ---------------------------------------------------------------------------
# Frontier metrics
# ---------------------------------------------------------------------------

def compute_frontier_metrics(score, frontier_pos, target_token_id, tokenizer, top_k=5):
    """
    Compute metrics at the frontier position from the score tensor.

    Args:
        score: [B, L, V] score tensor (probabilities, i.e. exp(log_score))
        frontier_pos: int, position of the frontier
        target_token_id: int, ground-truth token id for surprisal
        tokenizer: GPT2TokenizerFast for decoding token ids
        top_k: number of top candidates to return

    Returns:
        dict with surprisal, entropy, top_k_tokens, argmax_token
    """
    probs = score[0, frontier_pos]  # [V]
    probs_sum = probs.sum()
    p = probs / probs_sum.clamp(min=1e-30)

    log2_p = torch.log2(p.clamp(min=1e-30))

    target_prob = p[target_token_id].item()
    surprisal = -np.log2(max(target_prob, 1e-30))

    entropy = -(p * log2_p).sum().item()
    # Filter out NaN from zero-prob entries
    if np.isnan(entropy):
        entropy = 0.0

    topk_vals, topk_ids = probs.topk(top_k)
    topk_total = topk_vals.sum()
    top_k_tokens = []
    for val, tid in zip(topk_vals, topk_ids):
        tid_int = tid.item()
        token_str = tokenizer.decode([tid_int])
        prob = val.item() / probs_sum.item() if probs_sum.item() > 0 else 0.0
        top_k_tokens.append((tid_int, token_str, prob))

    argmax_id = probs.argmax().item()

    return {
        "surprisal": surprisal,
        "entropy": entropy,
        "target_prob": target_prob,
        "top_k": top_k_tokens,
        "argmax_id": argmax_id,
        "argmax_token": tokenizer.decode([argmax_id]),
    }


def compute_flickering(argmax_history):
    """
    Count how many times the argmax token changed across steps.

    Args:
        argmax_history: list of token ids (one per step)
    Returns:
        int: number of changes
    """
    changes = 0
    for i in range(1, len(argmax_history)):
        if argmax_history[i] != argmax_history[i - 1]:
            changes += 1
    return changes


# ---------------------------------------------------------------------------
# Step-state logging
# ---------------------------------------------------------------------------

def log_step_state(x, frontier, prefix_len, mask_token, tokenizer, max_display=20):
    """
    Print a compact view of the current sequence state.

    Shows up to max_display positions around the frontier, marking each as:
      [P] = prefix (fixed)
      [C] = committed (unmasked by LTR)
      [F] = frontier (current position being denoised)
      [M] = masked
    """
    seq = x[0].tolist()
    total = len(seq)

    start = max(0, frontier - 10)
    end = min(total, frontier + 10)

    parts = []
    for pos in range(start, end):
        tok = seq[pos]
        if pos < prefix_len:
            label = "P"
        elif pos == frontier:
            label = "F"
        elif tok == mask_token:
            label = "M"
        else:
            label = "C"

        if tok == mask_token:
            tok_str = "[MASK]"
        else:
            tok_str = repr(tokenizer.decode([tok]))

        parts.append(f"{pos}:{label}:{tok_str}")

    return " | ".join(parts)
