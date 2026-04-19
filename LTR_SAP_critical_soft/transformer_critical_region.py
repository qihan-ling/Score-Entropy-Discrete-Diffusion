"""
Sequential rescheduling with soft context for a 6-token critical window.

Each of the 6 positions (crit-2 through crit+3) gets a full 1024-step denoising
schedule.  After position k commits, a *soft embedding* is computed from the
sampler distribution blended with the ground-truth token (controlled by --lambda_val).
That soft embedding is injected into the model for all subsequent rounds.

Usage:
  python LTR_SAP_critical_soft/transformer_critical_region.py \
      --sentence "If the supervisor changes, the schedule deserves further inspection ." \
      --crit_word_pos 8 --lambda_val 1.0

See LTR_SAP_critical_soft/DESIGN.md for the full design document.
"""

import argparse
import json
import os
import sys
import re

import numpy as np
import torch
import torch.nn.functional as F

_this_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.join(_this_dir, "..")
sys.path.insert(0, _this_dir)
sys.path.append(_repo_root)

from sedd_helpers import (
    load_sedd_model,
    get_sampling_score_fn,
    tokenize_sentence,
    compute_kl_divergence,
)
from sampling import get_predictor, Denoiser
from soft_context_wrapper import SoftContextWrapper, build_soft_embedding


# ---------------------------------------------------------------------------
# Word-to-token mapping (same logic as transformer_critical_position.py)
# ---------------------------------------------------------------------------

def _word_to_token_breaks(words, sentence_tokens):
    """Map whitespace-delimited word indices to BPE token indices."""
    def _clean(token):
        return re.sub(r"[^a-zA-Z0-9*.,!?\\-]", "", token)

    cleaned = [_clean(t) for t in sentence_tokens]
    breaks = []
    idx_word = 0

    for idx_piece, piece in enumerate(cleaned):
        if idx_word >= len(words):
            break
        word = words[idx_word]
        if piece == word[:len(piece)]:
            breaks.append(idx_piece)
            idx_word += 1

    return breaks


# ---------------------------------------------------------------------------
# Per-step metric helpers
# ---------------------------------------------------------------------------

def _compute_raw_metrics(raw_score, pos, target_tok_id, prev_probs_raw):
    """Compute P_metric-based (raw score) metrics at one position.

    Returns (metrics_dict, current_probs_tensor).
    """
    probs = raw_score[0, pos]
    probs_sum = probs.sum()
    p = probs / probs_sum.clamp(min=1e-30)

    log2_p = torch.log2(p.clamp(min=1e-30))
    entropy = -(p * log2_p).sum().item()
    if np.isnan(entropy):
        entropy = 0.0

    p_target = p[target_tok_id].item() if target_tok_id is not None else None
    surprisal = -np.log2(max(p_target, 1e-30)) if p_target is not None else None

    topk_vals, topk_ids = probs.topk(min(5, probs.shape[0]))
    top5 = [(tid.item(), (val.item() / probs_sum.item()) if probs_sum.item() > 0 else 0.0)
            for val, tid in zip(topk_vals, topk_ids)]

    kl = 0.0
    if prev_probs_raw is not None:
        kl = compute_kl_divergence(p, prev_probs_raw)

    return {
        "entropy": entropy,
        "p_target": p_target,
        "surprisal": surprisal,
        "kl_from_prev": kl,
        "top5_ids": [t[0] for t in top5],
        "top5_probs": [t[1] for t in top5],
    }, p


def _compute_sampler_metrics(stag_score, transp_trans, pos, target_tok_id, graph,
                             prev_sampler_probs=None):
    """Compute P_sampler-based metrics at one position.

    Args:
        stag_score: [B, L, V] staggered score
        transp_trans: [B, L, V] transposed transition
        pos: frontier position
        target_tok_id: ground-truth token id
        graph: the absorbing graph
        prev_sampler_probs: previous step's sampler distribution for KL

    Returns (dict, sampler_probs_tensor).
    """
    probs = stag_score[0, pos] * transp_trans[0, pos]
    if graph.absorb:
        probs = probs[:-1]
    probs_sum = probs.sum()
    p = probs / probs_sum.clamp(min=1e-30)

    log2_p = torch.log2(p.clamp(min=1e-30))
    entropy = -(p * log2_p).sum().item()
    if np.isnan(entropy):
        entropy = 0.0

    p_target = p[target_tok_id].item() if target_tok_id is not None else None

    if prev_sampler_probs is not None:
        kl = compute_kl_divergence(p, prev_sampler_probs)
    else:
        kl = 0.0

    return {
        "sampler_entropy": entropy,
        "sampler_p_target": p_target,
        "sampler_kl_from_prev": kl,
    }, p


def _compute_commitment_details(stag_score, transp_trans, raw_score, pos, graph, vocab_size):
    """Compute top-50 details from both distributions at commitment time.

    Returns dict with top50_ids/probs (sampler) and p_model_top50_ids/probs (raw).
    Also returns the full sampler distribution for building soft context.
    """
    sampler_probs = stag_score[0, pos] * transp_trans[0, pos]
    if graph.absorb:
        sampler_probs = sampler_probs[:-1]
    sampler_sum = sampler_probs.sum().clamp(min=1e-30)
    sampler_p = sampler_probs / sampler_sum

    raw_probs = raw_score[0, pos]
    raw_sum = raw_probs.sum().clamp(min=1e-30)
    raw_p = raw_probs / raw_sum
    if raw_p.shape[0] > vocab_size:
        raw_p = raw_p[:vocab_size]
        raw_p = raw_p / raw_p.sum().clamp(min=1e-30)

    k = min(50, sampler_p.shape[0])
    s_vals, s_ids = sampler_p.topk(k)
    r_vals, r_ids = raw_p.topk(k)

    return {
        "top50_ids": s_ids.tolist(),
        "top50_probs": s_vals.tolist(),
        "p_model_top50_ids": r_ids.tolist(),
        "p_model_top50_probs": r_vals.tolist(),
    }, sampler_p


# ---------------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------------

def run_critical_region(
    sentence,
    crit_word_pos,
    lambda_val,
    steps,
    pad_length,
    device,
    seed=42,
    output_path=None,
    model_path="louaaron/sedd-medium",
    model_bundle=None,
    track_token_groups=False,
):
    """Run sequential rescheduling with soft context over a 6-token critical window.

    Args:
        sentence: full sentence string
        crit_word_pos: 1-indexed word position of the critical token
        lambda_val: soft-context lambda (0.0 = pure model, 1.0 = hard ground-truth)
        steps: denoising steps per position (e.g. 1024)
        pad_length: total sequence length (padded with MASK)
        device: torch device
        seed: random seed
        output_path: path to save JSON results
        model_path: HuggingFace model path
        model_bundle: optional (model, graph, noise, tokenizer)

    Returns:
        dict with per-position commitment_log, frontier_history, config
    """
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    if model_bundle is not None:
        model, graph, noise, tokenizer = model_bundle
    else:
        model, graph, noise, tokenizer = load_sedd_model(model_path, device)

    MASK = graph.dim - 1
    vocab_size = graph.dim - 1 if graph.absorb else graph.dim
    eps = 1e-5

    group_tracker = None
    if track_token_groups:
        sys.path.insert(0, os.path.join(_repo_root, "LTR_SAP_critical"))
        from token_group_utils import TokenGroupTracker
        group_tracker = TokenGroupTracker(tokenizer, model, device)

    full_ids = tokenize_sentence(tokenizer, sentence)
    words = sentence.split()
    sentence_tokens = tokenizer.tokenize(sentence)
    breaks = _word_to_token_breaks(words, sentence_tokens)

    # Determine the 6 target word positions: [crit-2, crit-1, crit, crit+1, crit+2, crit+3]
    window_offsets = [-2, -1, 0, 1, 2, 3]
    target_word_positions = []
    for offset in window_offsets:
        wpos = crit_word_pos + offset
        if wpos >= 1 and wpos <= len(words):
            target_word_positions.append((offset, wpos))

    if not target_word_positions:
        raise ValueError(f"No valid positions for crit_word_pos={crit_word_pos}, sentence has {len(words)} words")

    # Token position of the first word in the window determines the hard prefix boundary
    first_offset, first_wpos = target_word_positions[0]
    first_wpos_0idx = first_wpos - 1
    hard_prefix_end = breaks[first_wpos_0idx] + 1  # +1 for <|endoftext|>
    hard_prefix_ids = full_ids[:hard_prefix_end]

    print(f"  sentence:        {repr(sentence)}")
    print(f"  crit_word_pos:   {crit_word_pos} ({repr(words[crit_word_pos - 1])})")
    print(f"  window:          {[(o, words[wp - 1]) for o, wp in target_word_positions]}")
    print(f"  hard_prefix_end: token position {hard_prefix_end}")
    print(f"  lambda:          {lambda_val}")
    print(f"  pad_length:      {pad_length}")
    print()

    wrapper = SoftContextWrapper(model)
    score_fn = get_sampling_score_fn(wrapper)
    predictor = get_predictor("analytic")(graph, noise)

    embedding_table = model.vocab_embed.embedding.data

    soft_positions = []
    soft_embeddings = []

    all_results = []

    for round_idx, (offset, wpos) in enumerate(target_word_positions):
        wpos_0idx = wpos - 1
        frontier = breaks[wpos_0idx] + 1  # +1 for <|endoftext|>
        target_tok = full_ids[frontier] if frontier < len(full_ids) else None

        print(f"--- Round {round_idx + 1}/{len(target_word_positions)}: "
              f"offset={offset:+d}, word={repr(words[wpos_0idx])}, "
              f"tok_pos={frontier}, target_tok={repr(tokenizer.decode([target_tok])) if target_tok else 'N/A'} ---")

        has_soft = isinstance(soft_embeddings, torch.Tensor) and len(soft_positions) > 0
        wrapper.set_soft_context(soft_positions, soft_embeddings if has_soft else None)

        torch.manual_seed(seed + round_idx)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed + round_idx)

        x = torch.full((1, pad_length), MASK, dtype=torch.long, device=device)
        x[:, :len(hard_prefix_ids)] = torch.tensor(hard_prefix_ids, device=device)[None]

        for sp in soft_positions:
            gt_tok_at_sp = full_ids[sp]
            x[:, sp] = gt_tok_at_sp

        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        dt = (1 - eps) / steps

        frontier_history = []
        commitment_entry = None
        prev_probs_raw = None
        cumulative_kl = 0.0
        prev_sampler_probs = None
        sampler_cumulative_kl = 0.0

        with torch.no_grad():
            for i in range(steps):
                t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)

                # Re-enforce prefix and soft-context positions
                x[:, :len(hard_prefix_ids)] = torch.tensor(hard_prefix_ids, device=device)[None]
                for sp in soft_positions:
                    x[:, sp] = full_ids[sp]

                # LTR: mask everything after frontier
                if frontier + 1 < pad_length:
                    x[:, frontier + 1:] = MASK

                # Check if frontier already committed
                if x[0, frontier].item() != MASK:
                    committed_token = x[0, frontier].item()
                    committed_str = tokenizer.decode([committed_token])
                    correct = (committed_token == target_tok) if target_tok is not None else None

                    # Compute commitment-time details from the last step's score
                    commit_details = {}
                    sampler_dist = None
                    if frontier_history:
                        curr_sigma = noise(t)[0]
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            raw_score_commit = score_fn(x, curr_sigma)
                        stag = graph.staggered_score(raw_score_commit, curr_sigma)
                        tt = graph.transp_transition(x, curr_sigma)
                        commit_details, sampler_dist = _compute_commitment_details(
                            stag, tt, raw_score_commit, frontier, graph, vocab_size
                        )

                    commitment_entry = {
                        "position": frontier,
                        "word_position": wpos,
                        "word": words[wpos_0idx],
                        "offset": offset,
                        "step": i,
                        "steps_taken": i,
                        "t_commitment": timesteps[i].item(),
                        "committed_token_id": committed_token,
                        "committed_token": committed_str,
                        "final_surprisal": frontier_history[-1]["surprisal"] if frontier_history else None,
                        "final_entropy": frontier_history[-1]["entropy"] if frontier_history else None,
                        "final_sampler_entropy": frontier_history[-1].get("sampler_entropy") if frontier_history else None,
                        "final_sampler_p_target": frontier_history[-1].get("sampler_p_target") if frontier_history else None,
                        "cumulative_kl": cumulative_kl,
                        "sampler_cumulative_kl": sampler_cumulative_kl,
                        "target_token_id": target_tok,
                        "target_token": tokenizer.decode([target_tok]) if target_tok else None,
                        "correct": correct,
                        **commit_details,
                    }
                    print(f"  COMMITTED at step {i}: {repr(committed_str)} "
                          f"(target: {repr(tokenizer.decode([target_tok])) if target_tok else 'N/A'}, "
                          f"correct: {correct})")
                    break

                # Compute metrics
                curr_sigma = noise(t)[0]
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    raw_score = score_fn(x, curr_sigma)

                raw_metrics, current_probs = _compute_raw_metrics(
                    raw_score, frontier, target_tok, prev_probs_raw
                )
                cumulative_kl += raw_metrics["kl_from_prev"]
                raw_metrics["cumulative_kl"] = cumulative_kl
                prev_probs_raw = current_probs

                stag_score = graph.staggered_score(raw_score, curr_sigma)
                transp_trans = graph.transp_transition(x, curr_sigma)
                sampler_metrics, current_sampler_probs = _compute_sampler_metrics(
                    stag_score, transp_trans, frontier, target_tok, graph,
                    prev_sampler_probs=prev_sampler_probs,
                )
                sampler_cumulative_kl += sampler_metrics["sampler_kl_from_prev"]
                sampler_metrics["sampler_cumulative_kl"] = sampler_cumulative_kl
                prev_sampler_probs = current_sampler_probs

                group_metrics = {}
                if group_tracker is not None:
                    group_metrics = group_tracker.compute_group_metrics(current_probs, target_tok)

                frontier_history.append({
                    "step": i,
                    "t": timesteps[i].item(),
                    **raw_metrics,
                    **sampler_metrics,
                    **group_metrics,
                })

                # Core predictor step
                x = predictor.update_fn(score_fn, x, t, dt)

                # Re-enforce prefix, soft positions, and LTR
                x[:, :len(hard_prefix_ids)] = torch.tensor(hard_prefix_ids, device=device)[None]
                for sp in soft_positions:
                    x[:, sp] = full_ids[sp]
                if frontier + 1 < pad_length:
                    x[:, frontier + 1:] = MASK

                # Periodic logging
                if i < 3 or i % max(1, steps // 20) == 0 or i == steps - 1:
                    tok_val = x[0, frontier].item()
                    tok_str = "[MASK]" if tok_val == MASK else repr(tokenizer.decode([tok_val]))
                    print(f"  step {i:5d} | t={timesteps[i].item():.4f} | frontier={tok_str}")

        # If never committed, run final denoiser
        if commitment_entry is None:
            print(f"  Running final denoiser step...")
            denoiser = Denoiser(graph, noise)
            x[:, :len(hard_prefix_ids)] = torch.tensor(hard_prefix_ids, device=device)[None]
            for sp in soft_positions:
                x[:, sp] = full_ids[sp]
            if frontier + 1 < pad_length:
                x[:, frontier + 1:] = MASK

            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
            x = denoiser.update_fn(score_fn, x, t)
            x[:, :len(hard_prefix_ids)] = torch.tensor(hard_prefix_ids, device=device)[None]

            committed_token = x[0, frontier].item()
            committed_str = tokenizer.decode([committed_token])
            correct = (committed_token == target_tok) if target_tok is not None else None

            commitment_entry = {
                "position": frontier,
                "word_position": wpos,
                "word": words[wpos_0idx],
                "offset": offset,
                "step": steps,
                "steps_taken": steps,
                "t_commitment": timesteps[-1].item(),
                "committed_token_id": committed_token,
                "committed_token": committed_str,
                "final_surprisal": frontier_history[-1]["surprisal"] if frontier_history else None,
                "final_entropy": frontier_history[-1]["entropy"] if frontier_history else None,
                "final_sampler_entropy": frontier_history[-1].get("sampler_entropy") if frontier_history else None,
                "final_sampler_p_target": frontier_history[-1].get("sampler_p_target") if frontier_history else None,
                "cumulative_kl": cumulative_kl,
                "sampler_cumulative_kl": sampler_cumulative_kl,
                "target_token_id": target_tok,
                "target_token": tokenizer.decode([target_tok]) if target_tok else None,
                "correct": correct,
            }
            print(f"  COMMITTED (final denoiser): {repr(committed_str)}")

        # Build soft embedding for the next round
        # Use sampler distribution at the commitment step if available,
        # otherwise fall back to one-hot ground truth
        if target_tok is not None:
            if commitment_entry.get("top50_ids") is not None:
                # Reconstruct full sampler distribution from commitment details
                sampler_full = torch.zeros(vocab_size, device=device)
                for tid, tprob in zip(commitment_entry["top50_ids"], commitment_entry["top50_probs"]):
                    if tid < vocab_size:
                        sampler_full[tid] = tprob
                sampler_full = sampler_full / sampler_full.sum().clamp(min=1e-30)
                p_model = sampler_full
            else:
                p_model = F.one_hot(torch.tensor(target_tok), num_classes=vocab_size).float().to(device)

            gt_one_hot = F.one_hot(torch.tensor(target_tok), num_classes=vocab_size).float().to(device)
            p_soft = (1 - lambda_val) * p_model + lambda_val * gt_one_hot
            e_soft = build_soft_embedding(p_soft, embedding_table, vocab_size)

            soft_positions.append(frontier)
            if len(soft_embeddings) == 0:
                soft_embeddings = e_soft.unsqueeze(0)
            else:
                soft_embeddings = torch.cat([soft_embeddings, e_soft.unsqueeze(0)], dim=0)

        # Store results for this round
        all_results.append({
            "offset": offset,
            "word_position": wpos,
            "word": words[wpos_0idx],
            "token_position": frontier,
            "target_token_id": target_tok,
            "target_token": tokenizer.decode([target_tok]) if target_tok else None,
            "commitment": commitment_entry,
            "frontier_history": frontier_history,
        })

        print()

    # Assemble output
    result = {
        "config": {
            "model_path": model_path,
            "steps": steps,
            "seed": seed,
            "lambda_val": lambda_val,
            "pad_length": pad_length,
            "experiment_type": "critical_region_soft_context",
            "crit_word_pos": crit_word_pos,
            "window_offsets": [o for o, _ in target_word_positions],
            "track_token_groups": track_token_groups,
        },
        "tokenization": {
            "full_ids": full_ids,
            "sentence": sentence,
            "sentence_length": len(full_ids),
            "hard_prefix_end": hard_prefix_end,
        },
        "positions": all_results,
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Results saved to {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Sequential rescheduling with soft context — 6-token critical window"
    )
    parser.add_argument("--model_path", type=str, default="louaaron/sedd-medium")
    parser.add_argument("--sentence", type=str, required=True)
    parser.add_argument(
        "--crit_word_pos", type=int, required=True,
        help="1-indexed word position of the critical token (center of 6-token window)",
    )
    parser.add_argument("--lambda_val", type=float, default=1.0,
                        help="Soft context lambda: 0.0=pure model, 1.0=hard ground-truth")
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--pad_length", type=int, default=256,
                        help="Total sequence length (padded with MASK)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--track_token_groups", action="store_true",
                        help="Track syntactic/semantic group probabilities per step")

    args = parser.parse_args()
    device = torch.device(args.device)

    run_critical_region(
        sentence=args.sentence,
        crit_word_pos=args.crit_word_pos,
        lambda_val=args.lambda_val,
        steps=args.steps,
        pad_length=args.pad_length,
        device=device,
        seed=args.seed,
        output_path=args.output_path,
        model_path=args.model_path,
        track_token_groups=args.track_token_groups,
    )


if __name__ == "__main__":
    main()
