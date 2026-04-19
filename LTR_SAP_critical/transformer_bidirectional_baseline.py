"""
Bidirectional-context baseline: cloze-style denoising for one target position.

All sentence tokens are revealed EXCEPT the target position, which starts as MASK.
The model runs the full 1024-step denoising schedule to unmask just that one position.

This isolates position-level linguistic difficulty from the effect of limited context.
Comparison with the unidirectional critical-position results reveals the "value of
right context".

Input construction for target position k:
  [<eot>, tok_1, ..., tok_{k-1}, MASK, tok_{k+1}, ..., tok_N, MASK_pad, ..., MASK_pad]

- All sentence positions (except target): ground-truth tokens (hard)
- Pad to fixed length (default 256) with MASK
- Only the target position can change; sentence context and padding stay fixed

Usage:
  python LTR_SAP_critical/transformer_bidirectional_baseline.py \
      --sentence "If the supervisor changes, the schedule deserves further inspection ." \
      --target_position 7 \
      --output_path LTR_SAP_critical/bidirectional/test.json
"""

import argparse
import json
import os
import re
import sys

import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sedd_helpers import (
    load_sedd_model,
    get_sampling_score_fn,
    tokenize_sentence,
    compute_frontier_metrics,
    compute_kl_divergence,
)
from sampling import get_predictor, Denoiser
import torch.nn.functional as F


def run_bidirectional_baseline(
    sentence,
    target_word_position,
    steps,
    device,
    seed=42,
    output_path=None,
    model_path="louaaron/sedd-medium",
    model_bundle=None,
    pad_length=256,
    track_prefix_scores=False,
    track_token_groups=False,
):
    """Run a bidirectional cloze-style denoising pass for one target position.

    Args:
        sentence: full sentence string
        target_word_position: 1-indexed word position to target
        steps: number of denoising steps
        device: torch device
        seed: random seed
        output_path: path to save JSON results
        model_path: HuggingFace model path
        model_bundle: optional pre-loaded (model, graph, noise, tokenizer)
        pad_length: total sequence length to pad to
        track_prefix_scores: log p(gt) at sentence positions to measure context usage
    """
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    if model_bundle is not None:
        model, graph, noise, tokenizer = model_bundle
    else:
        model, graph, noise, tokenizer = load_sedd_model(model_path, device)

    MASK = graph.dim - 1
    eps = 1e-5

    full_ids = tokenize_sentence(tokenizer, sentence)
    words = sentence.split()

    # Word-to-token mapping
    sentence_tokens = tokenizer.tokenize(sentence)

    def _clean(token):
        return re.sub(r"[^a-zA-Z0-9*.,!?\\-]", "", token)

    cleaned = [_clean(t) for t in sentence_tokens]
    breaks = []
    idx_word = 0

    for idx_piece, piece in enumerate(cleaned):
        if idx_word < len(words):
            word = words[idx_word]
        else:
            break
        if piece == word[:len(piece)]:
            breaks.append(idx_piece)
            idx_word += 1

    target_word_0idx = target_word_position - 1
    if target_word_0idx < 0 or target_word_0idx >= len(breaks):
        raise ValueError(
            f"target_word_position {target_word_position} out of range "
            f"(sentence has {len(words)} words)"
        )

    frontier = breaks[target_word_0idx] + 1  # +1 for <|endoftext|>
    sentence_end = len(full_ids)
    target_tok = full_ids[frontier] if frontier < len(full_ids) else None

    # Build the bidirectional input:
    #   [<eot>, tok_1, ..., MASK, tok_{k+1}, ..., tok_N, MASK_pad, ..., MASK_pad]
    bidir_ids = list(full_ids)
    bidir_ids[frontier] = MASK

    # Pad to pad_length
    if len(bidir_ids) < pad_length:
        bidir_ids.extend([MASK] * (pad_length - len(bidir_ids)))
    else:
        bidir_ids = bidir_ids[:pad_length]

    # Positions that must stay fixed: all sentence positions except target, plus padding
    fixed_sentence = list(range(0, frontier)) + list(range(frontier + 1, sentence_end))

    print(f"  sentence:        {repr(sentence)}")
    print(f"  target word:     {repr(words[target_word_0idx])} (word_pos={target_word_position})")
    print(f"  target position: token position {frontier}")
    print(f"  target_token:    {repr(tokenizer.decode([target_tok])) if target_tok else 'N/A'}")
    print(f"  sentence_end:    {sentence_end} tokens")
    print(f"  pad_length:      {pad_length}")
    print(f"  experiment:      bidirectional_baseline")
    print()

    group_tracker = None
    if track_token_groups:
        from token_group_utils import get_or_create_tracker
        group_tracker = get_or_create_tracker(tokenizer, model, device)

    score_fn = get_sampling_score_fn(model)
    predictor = get_predictor("analytic")(graph, noise)

    x = torch.tensor([bidir_ids], dtype=torch.long, device=device)
    timesteps = torch.linspace(1, eps, steps + 1, device=device)
    dt = (1 - eps) / steps

    frontier_history = []
    prefix_scores_log = []
    commitment_entry = None
    prev_probs = None
    cumulative_kl = 0.0
    prev_sampler_probs = None
    sampler_cumulative_kl = 0.0

    print(f"Sampling loop ({steps} steps, target at position {frontier})...\n")

    with torch.no_grad():
        for i in range(steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)

            # Re-enforce context: fix all sentence positions except target
            for pos in fixed_sentence:
                if pos < len(full_ids):
                    x[:, pos] = full_ids[pos]
            # Fix padding
            x[:, sentence_end:] = MASK

            # Check if target already committed
            if x[0, frontier].item() != MASK:
                committed_token = x[0, frontier].item()
                committed_str = tokenizer.decode([committed_token])
                correct = (committed_token == target_tok) if target_tok is not None else None

                commit_extras = {}
                curr_sigma_c = noise(t)[0]
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    raw_score_c = score_fn(x, curr_sigma_c)

                # Sampler distribution at commitment
                stag_c = graph.staggered_score(raw_score_c, curr_sigma_c)
                tt_c = graph.transp_transition(x, curr_sigma_c)
                s_probs = stag_c[0, frontier] * tt_c[0, frontier]
                if graph.absorb:
                    s_probs = s_probs[:-1]
                s_sum = s_probs.sum().clamp(min=1e-30)
                s_p = s_probs / s_sum
                k50 = min(50, s_p.shape[0])
                s_vals, s_ids = s_p.topk(k50)
                commit_extras["top50_ids"] = s_ids.tolist()
                commit_extras["top50_probs"] = s_vals.tolist()

                # Raw score distribution at commitment
                r_probs = raw_score_c[0, frontier]
                r_sum = r_probs.sum().clamp(min=1e-30)
                r_p = r_probs / r_sum
                vocab_size = graph.dim - 1 if graph.absorb else graph.dim
                if r_p.shape[0] > vocab_size:
                    r_p = r_p[:vocab_size]
                    r_p = r_p / r_p.sum().clamp(min=1e-30)
                r_vals, r_ids = r_p.topk(min(50, r_p.shape[0]))
                commit_extras["p_model_top50_ids"] = r_ids.tolist()
                commit_extras["p_model_top50_probs"] = r_vals.tolist()

                commitment_entry = {
                    "position": frontier,
                    "word_position": target_word_position,
                    "word": words[target_word_0idx],
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
                    **commit_extras,
                }
                print(f"  COMMITTED at step {i}: {repr(committed_str)} "
                      f"(target: {repr(tokenizer.decode([target_tok])) if target_tok else 'N/A'}, "
                      f"correct: {correct})")
                break

            # Compute scores at target position
            curr_sigma = noise(t)[0]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                raw_score = score_fn(x, curr_sigma)

            if target_tok is not None:
                metrics = compute_frontier_metrics(raw_score, frontier, target_tok, tokenizer)
            else:
                metrics = compute_frontier_metrics(raw_score, frontier, 0, tokenizer)
                metrics["surprisal"] = None
                metrics["target_prob"] = None

            current_probs = metrics.pop("probs")

            if prev_probs is not None:
                step_kl = compute_kl_divergence(current_probs, prev_probs)
            else:
                step_kl = 0.0
            cumulative_kl += step_kl
            metrics["kl_from_prev"] = step_kl
            metrics["cumulative_kl"] = cumulative_kl
            prev_probs = current_probs

            metrics["p_target"] = metrics.pop("target_prob", None)
            if metrics.get("top_k"):
                metrics["top5_ids"] = [t[0] for t in metrics["top_k"]]
                metrics["top5_probs"] = [t[2] for t in metrics["top_k"]]

            # Sampler-based metrics
            stag_score = graph.staggered_score(raw_score, curr_sigma)
            transp_trans = graph.transp_transition(x, curr_sigma)
            sampler_probs = stag_score[0, frontier] * transp_trans[0, frontier]
            if graph.absorb:
                sampler_probs = sampler_probs[:-1]
            sampler_sum = sampler_probs.sum().clamp(min=1e-30)
            sp = sampler_probs / sampler_sum
            sp_log2 = torch.log2(sp.clamp(min=1e-30))
            sampler_entropy = -(sp * sp_log2).sum().item()
            if np.isnan(sampler_entropy):
                sampler_entropy = 0.0
            sampler_p_target = sp[target_tok].item() if target_tok is not None else None
            metrics["sampler_entropy"] = sampler_entropy
            metrics["sampler_p_target"] = sampler_p_target

            if prev_sampler_probs is not None:
                sampler_step_kl = compute_kl_divergence(sp, prev_sampler_probs)
            else:
                sampler_step_kl = 0.0
            sampler_cumulative_kl += sampler_step_kl
            metrics["sampler_kl_from_prev"] = sampler_step_kl
            metrics["sampler_cumulative_kl"] = sampler_cumulative_kl
            prev_sampler_probs = sp

            if group_tracker is not None:
                group_metrics = group_tracker.compute_group_metrics(current_probs, target_tok)
                metrics.update(group_metrics)

            frontier_history.append({
                "step": i,
                "t": timesteps[i].item(),
                **metrics,
            })

            # Prefix score probing
            if track_prefix_scores and i % max(1, steps // 100) == 0:
                prefix_entry = {"step": i, "t": timesteps[i].item(), "positions": {}}
                for ppos in fixed_sentence:
                    if ppos >= sentence_end or ppos == 0:
                        continue
                    gt_tok = full_ids[ppos]
                    p_probs = raw_score[0, ppos]
                    p_sum = p_probs.sum().clamp(min=1e-30)
                    p_gt = (p_probs[gt_tok] / p_sum).item()
                    prefix_entry["positions"][str(ppos)] = {
                        "token_id": gt_tok,
                        "p_gt": p_gt,
                    }
                prefix_scores_log.append(prefix_entry)

            # Core predictor step
            x = predictor.update_fn(score_fn, x, t, dt)

            # Re-enforce context and padding after predictor
            for pos in fixed_sentence:
                if pos < len(full_ids):
                    x[:, pos] = full_ids[pos]
            x[:, sentence_end:] = MASK

            # Periodic logging
            if i < 5 or i % max(1, steps // 20) == 0 or i == steps - 1:
                tok_val = x[0, frontier].item()
                tok_str = "[MASK]" if tok_val == MASK else repr(tokenizer.decode([tok_val]))
                print(f"  step {i:5d} | t={timesteps[i].item():.4f} | target={tok_str}")

    # Final denoiser if not committed during loop
    if commitment_entry is None:
        print("\n  Running final denoiser step...")
        denoiser = Denoiser(graph, noise)
        t_final = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)

        for pos in fixed_sentence:
            if pos < len(full_ids):
                x[:, pos] = full_ids[pos]
        x[:, sentence_end:] = MASK

        x = denoiser.update_fn(score_fn, x, t_final, dt)

        for pos in fixed_sentence:
            if pos < len(full_ids):
                x[:, pos] = full_ids[pos]
        x[:, sentence_end:] = MASK

        committed_token = x[0, frontier].item()
        committed_str = tokenizer.decode([committed_token])
        correct = (committed_token == target_tok) if target_tok is not None else None

        commitment_entry = {
            "position": frontier,
            "word_position": target_word_position,
            "word": words[target_word_0idx],
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

    # Serialize frontier_history
    serializable_history = []
    for h in frontier_history:
        sh = {k: v for k, v in h.items() if k != "probs"}
        if sh.get("top_k"):
            sh["top_k"] = [
                {"id": t[0], "token": t[1], "prob": t[2]}
                if isinstance(t, (list, tuple)) else t
                for t in sh["top_k"]
            ]
        serializable_history.append(sh)

    if commitment_entry.get("target_prob") is not None:
        commitment_entry["p_target"] = commitment_entry.pop("target_prob")

    result = {
        "config": {
            "model_path": model_path,
            "steps": steps,
            "seed": seed,
            "experiment_type": "bidirectional_baseline",
            "target_word_position": target_word_position,
            "pad_length": pad_length,
            "track_prefix_scores": track_prefix_scores,
            "track_token_groups": track_token_groups,
        },
        "tokenization": {
            "full_ids": full_ids,
            "sentence": sentence,
            "sentence_length": len(full_ids),
            "frontier": frontier,
        },
        "commitment_log": commitment_entry,
        "frontier_history": serializable_history,
    }
    if track_prefix_scores and prefix_scores_log:
        result["prefix_scores_log"] = prefix_scores_log

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Bidirectional baseline: cloze-style denoising with full sentence context"
    )
    parser.add_argument("--model_path", type=str, default="louaaron/sedd-medium")
    parser.add_argument("--sentence", type=str, required=True)
    parser.add_argument(
        "--target_position", type=int, required=True,
        help="1-indexed word position to target",
    )
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--pad_length", type=int, default=256)
    parser.add_argument("--track_prefix_scores", action="store_true")
    parser.add_argument("--track_token_groups", action="store_true",
                        help="Track syntactic/semantic group probabilities per step")

    args = parser.parse_args()
    device = torch.device(args.device)

    run_bidirectional_baseline(
        sentence=args.sentence,
        target_word_position=args.target_position,
        steps=args.steps,
        device=device,
        seed=args.seed,
        output_path=args.output_path,
        model_path=args.model_path,
        pad_length=args.pad_length,
        track_prefix_scores=args.track_prefix_scores,
        track_token_groups=args.track_token_groups,
    )


if __name__ == "__main__":
    main()
