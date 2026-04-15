"""
Critical-position denoising: run a single denoising pass targeting one position.

Unlike strict-LTR which processes all positions sequentially (causing a position
confound where later positions get fewer steps), this script gives each target
position the FULL noise schedule (all --steps steps from t=1 to t=eps).

Input construction for target position k:
  [<|endoftext|>, token_1, ..., token_{k-1}, MASK, MASK, ..., MASK]

All tokens before k are given as correct prefix. Position k denoises from step 0.
LTR is enforced (future positions stay MASK), but we also record the model's
score distribution at future positions (soft view without commitment).

Usage:
  python LTR_SAP_critical/transformer_critical_position.py \
      --sentence "If the supervisor changes, the schedule deserves further inspection ." \
      --target_position 7 \
      --output_path LTR_SAP_critical/test.json
"""

import argparse
import json
import os
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


def run_critical_position(
    sentence,
    target_word_position,
    steps,
    device,
    seed=42,
    output_path=None,
    model_path="louaaron/sedd-medium",
    model_bundle=None,
    future_topk=5,
    future_window=3,
):
    """Run a single critical-position denoising pass.

    Args:
        sentence: full sentence string
        target_word_position: 1-indexed word position to target (matches SAP CSV convention)
        steps: number of denoising steps (e.g. 1024)
        device: torch device
        seed: random seed
        output_path: path to save JSON results
        model_path: HuggingFace model path
        model_bundle: optional (model, graph, noise, tokenizer) to avoid reloading
        future_topk: number of top-K tokens to track at future positions
        future_window: how many future positions to track scores for

    Returns:
        dict with commitment_log, frontier_history, future_scores
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

    # Convert 1-indexed word position to token position
    # Token position 0 = <|endoftext|>, token positions 1..N = sentence tokens
    # We need to find which token(s) correspond to word at target_word_position
    sentence_tokens = tokenizer.tokenize(sentence)

    # Build word-to-token mapping
    import re
    def _clean(token):
        return re.sub(r"[^a-zA-Z0-9*.,!?\\-]", "", token)

    cleaned = [_clean(t) for t in sentence_tokens]
    breaks = []
    idx_word = 0
    current_pieces = []

    for idx_piece, piece in enumerate(cleaned):
        if idx_word < len(words):
            word = words[idx_word]
        else:
            break
        if piece == word[:len(piece)]:
            breaks.append(idx_piece)
            idx_word += 1

    # breaks[i] = token index (0-indexed within sentence_tokens) of word i's first token
    # Token position in full_ids = breaks[i] + 1 (for <|endoftext|> prefix)
    target_word_0idx = target_word_position - 1  # convert to 0-indexed
    if target_word_0idx < 0 or target_word_0idx >= len(breaks):
        raise ValueError(
            f"target_word_position {target_word_position} out of range "
            f"(sentence has {len(words)} words)"
        )

    # First token of the target word in full_ids
    frontier_start = breaks[target_word_0idx] + 1  # +1 for <|endoftext|>

    # Build prefix: all tokens before the target word's first token
    prefix_ids = full_ids[:frontier_start]
    prefix_len = len(prefix_ids)

    # Target token at the frontier position
    target_tok = full_ids[frontier_start] if frontier_start < len(full_ids) else None
    sentence_end = len(full_ids)

    # Determine future positions to track (within sentence bounds)
    future_positions = []
    for offset in range(1, future_window + 1):
        fpos = frontier_start + offset
        if fpos < sentence_end:
            future_positions.append(fpos)

    print(f"  sentence:        {repr(sentence)}")
    print(f"  target word:     {repr(words[target_word_0idx])} (word_pos={target_word_position})")
    print(f"  frontier_start:  token position {frontier_start}")
    print(f"  prefix_len:      {prefix_len} tokens")
    print(f"  target_token:    {repr(tokenizer.decode([target_tok])) if target_tok else 'N/A'}")
    print(f"  future tracking: positions {future_positions}")
    print()

    score_fn = get_sampling_score_fn(model)
    predictor = get_predictor("analytic")(graph, noise)

    x = graph.sample_limit(1, 1024).to(device)
    timesteps = torch.linspace(1, eps, steps + 1, device=device)
    dt = (1 - eps) / steps

    frontier = frontier_start

    frontier_history = []
    future_scores_log = []
    commitment_entry = None
    prev_probs = None
    cumulative_kl = 0.0

    print(f"Sampling loop ({steps} steps, frontier at position {frontier})...\n")

    with torch.no_grad():
        for i in range(steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)

            # Fix prefix tokens
            x[:, :prefix_len] = torch.tensor(prefix_ids, device=device)[None]

            # LTR enforcement: mask everything after frontier
            if frontier < 1024:
                x[:, frontier + 1:] = MASK

            # Check if frontier already committed
            if x[0, frontier].item() != MASK:
                committed_token = x[0, frontier].item()
                committed_str = tokenizer.decode([committed_token])
                correct = (committed_token == target_tok) if target_tok is not None else None

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
                    "cumulative_kl": cumulative_kl,
                    "target_token_id": target_tok,
                    "target_token": tokenizer.decode([target_tok]) if target_tok else None,
                    "correct": correct,
                }
                print(f"  COMMITTED at step {i}: {repr(committed_str)} "
                      f"(target: {repr(tokenizer.decode([target_tok])) if target_tok else 'N/A'}, "
                      f"correct: {correct})")
                break

            # Compute scores at frontier
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

            frontier_history.append({
                "step": i,
                "t": timesteps[i].item(),
                **metrics,
            })

            # Track future position scores (soft view)
            if future_positions and i % max(1, steps // 100) == 0:
                future_step_entry = {"step": i, "t": timesteps[i].item(), "positions": {}}
                for fpos in future_positions:
                    fpos_target = full_ids[fpos] if fpos < len(full_ids) else None
                    if fpos_target is not None:
                        fprobs = raw_score[0, fpos]
                        fprobs_sum = fprobs.sum()
                        fp = fprobs / fprobs_sum.clamp(min=1e-30)

                        ftopk_vals, ftopk_ids = fprobs.topk(min(future_topk, fprobs.shape[0]))
                        ftop_k = []
                        for val, tid in zip(ftopk_vals, ftopk_ids):
                            prob = val.item() / fprobs_sum.item() if fprobs_sum.item() > 0 else 0.0
                            ftop_k.append({
                                "id": tid.item(),
                                "token": tokenizer.decode([tid.item()]),
                                "prob": prob,
                            })

                        target_p = fp[fpos_target].item()
                        future_step_entry["positions"][str(fpos)] = {
                            "target_token_id": fpos_target,
                            "target_token": tokenizer.decode([fpos_target]),
                            "target_prob": target_p,
                            "surprisal": -np.log2(max(target_p, 1e-30)),
                            "entropy": -(fp * torch.log2(fp.clamp(min=1e-30))).sum().item(),
                            "top_k": ftop_k,
                        }
                future_scores_log.append(future_step_entry)

            # Core predictor step
            x = predictor.update_fn(score_fn, x, t, dt)

            # Re-enforce prefix and LTR
            x[:, :prefix_len] = torch.tensor(prefix_ids, device=device)[None]
            if frontier < 1024:
                x[:, frontier + 1:] = MASK

            # Periodic logging
            if i < 5 or i % max(1, steps // 20) == 0 or i == steps - 1:
                tok_val = x[0, frontier].item()
                tok_str = "[MASK]" if tok_val == MASK else repr(tokenizer.decode([tok_val]))
                print(f"  step {i:5d} | t={timesteps[i].item():.4f} | frontier={tok_str}")

    # If never committed during the loop, run final denoiser
    if commitment_entry is None:
        print(f"\n  Running final denoiser step...")
        denoiser = Denoiser(graph, noise)
        x[:, :prefix_len] = torch.tensor(prefix_ids, device=device)[None]
        if frontier < 1024:
            x[:, frontier + 1:] = MASK
        t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
        x = denoiser.update_fn(score_fn, x, t)
        x[:, :prefix_len] = torch.tensor(prefix_ids, device=device)[None]

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
            "cumulative_kl": cumulative_kl,
            "target_token_id": target_tok,
            "target_token": tokenizer.decode([target_tok]) if target_tok else None,
            "correct": correct,
        }
        print(f"  COMMITTED (final denoiser): {repr(committed_str)}")

    # Serialize frontier_history (remove probs, format top_k)
    serializable_history = []
    for h in frontier_history:
        sh = {k: v for k, v in h.items() if k != "probs"}
        if sh.get("top_k"):
            sh["top_k"] = [{"id": t[0], "token": t[1], "prob": t[2]} for t in sh["top_k"]]
        serializable_history.append(sh)

    result = {
        "config": {
            "model_path": model_path,
            "steps": steps,
            "seed": seed,
            "experiment_type": "critical_position",
            "target_word_position": target_word_position,
            "future_window": future_window,
        },
        "tokenization": {
            "full_ids": full_ids,
            "sentence": sentence,
            "sentence_length": len(full_ids),
            "prefix_len": prefix_len,
            "frontier_start": frontier_start,
        },
        "commitment_log": commitment_entry,
        "frontier_history": serializable_history,
        "future_scores": future_scores_log,
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Critical-position denoising: full noise schedule for one target position"
    )
    parser.add_argument("--model_path", type=str, default="louaaron/sedd-medium")
    parser.add_argument("--sentence", type=str, required=True)
    parser.add_argument(
        "--target_position", type=int, required=True,
        help="1-indexed word position to target",
    )
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--future_window", type=int, default=3)
    parser.add_argument("--future_topk", type=int, default=5)

    args = parser.parse_args()
    device = torch.device(args.device)

    run_critical_position(
        sentence=args.sentence,
        target_word_position=args.target_position,
        steps=args.steps,
        device=device,
        seed=args.seed,
        output_path=args.output_path,
        model_path=args.model_path,
        future_topk=args.future_topk,
        future_window=args.future_window,
    )


if __name__ == "__main__":
    main()
