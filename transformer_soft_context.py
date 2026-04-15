"""
Soft-context strict-LTR: use probability distributions as context instead of
discrete tokens at committed positions.

At each committed position, instead of feeding the hard discrete token, this
script feeds the probability distribution from the step when that position
committed, applied as a weighted sum over the embedding table:
  soft_embedding = probs @ model.vocab_embed.embedding

This is an out-of-distribution input (model was trained on discrete tokens),
so the primary purpose is to test whether soft distributional context carries
any useful signal.

Usage:
  python transformer_soft_context.py --enforce_prefix \
      --sentence "The horse raced past the barn fell ." \
      --output_path LTR_SAP_soft/test.json
"""

import argparse
import json
import os
import torch
import numpy as np

from sedd_helpers import (
    load_sedd_model,
    get_sampling_score_fn,
    set_causal_mode,
    compute_frontier_metrics,
    compute_kl_divergence,
    tokenize_sentence,
    log_step_state,
)
from sampling import get_predictor, Denoiser


def soft_context_forward(model, x, sigma, context_probs, MASK):
    """Forward pass with soft embeddings at committed positions.

    For positions with stored probability distributions, compute
    probs @ embedding_weight instead of a discrete lookup. For MASK
    and frontier positions, use the standard discrete embedding.
    """
    embedding_weight = model.vocab_embed.embedding  # [V, D]

    # Standard discrete embedding for all positions
    x_embed = embedding_weight[x[0]]  # [L, D]

    # Override committed positions with soft embeddings
    for pos in range(x.shape[1]):
        if pos in context_probs:
            probs = context_probs[pos]  # [V]
            x_embed[pos] = probs @ embedding_weight  # [D]

    x_embed = x_embed.unsqueeze(0)  # [1, L, D]

    # Run the rest of the model (bypass vocab_embed, go straight to transformer)
    c = torch.nn.functional.silu(model.sigma_map(sigma))
    rotary_cos_sin = model.rotary_emb(x_embed)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        h = x_embed
        for block in model.blocks:
            h = block(h, rotary_cos_sin, c, seqlens=None)
        h = model.output_layer(h, c)

    if model.scale_by_sigma:
        esigm1_log = torch.where(
            sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1
        ).log().to(h.dtype)[:, None, None]
        h = h - esigm1_log - np.log(h.shape[-1] - 1)

    # Zero out the score at the input token for each position
    h = torch.scatter(h, -1, x[..., None], torch.zeros_like(h[..., :1]))

    return h


def run_soft_context_experiment(
    model_path,
    sentence,
    steps,
    device,
    seed=42,
    output_path=None,
    model_bundle=None,
):
    """Run enforce-prefix strict-LTR with soft context at committed positions."""
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
    prefix_len = 1
    target_ids = full_ids
    sentence_end = len(full_ids)

    print(f"{'='*70}")
    print(f"SEDD Soft-Context Enforce-Prefix LTR")
    print(f"{'='*70}")
    print(f"  sentence:      {repr(sentence)}")
    print(f"  full_ids ({len(full_ids)}): {full_ids[:20]}")
    print()

    score_fn = get_sampling_score_fn(model)
    predictor = get_predictor("analytic")(graph, noise)
    denoiser = Denoiser(graph, noise)

    x = graph.sample_limit(1, 1024).to(device)
    timesteps = torch.linspace(1, eps, steps + 1, device=device)
    dt = (1 - eps) / steps

    frontier = prefix_len

    # Storage for soft context distributions
    context_probs = {}  # pos -> [V] tensor of probabilities at commitment

    frontier_history = {}
    commitment_log = []
    prev_probs = None
    cumulative_kl = 0.0
    current_frontier_start_step = 0

    print(f"Sampling loop ({steps} steps)...\n")

    with torch.no_grad():
        for i in range(steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)

            # Fix prefix (position 0 = <|endoftext|>)
            x[:, 0] = full_ids[0]

            # For committed positions (1..frontier-1), we still need discrete
            # tokens in x for the predictor step, but the model's score is
            # computed using soft embeddings
            committed_so_far = min(frontier, sentence_end)
            for pos in range(1, committed_so_far):
                x[:, pos] = target_ids[pos]

            # LTR enforcement
            if frontier < 1024:
                x[:, frontier + 1:] = MASK

            if frontier >= sentence_end:
                break

            # Compute scores using soft-context forward
            curr_sigma = noise(t)[0]
            raw_score = soft_context_forward(model, x, curr_sigma, context_probs, MASK)
            raw_score = raw_score.exp()  # log-score -> score (probabilities)

            target_tok = target_ids[frontier] if frontier < len(target_ids) else None
            if target_tok is not None:
                metrics = compute_frontier_metrics(
                    raw_score, frontier, target_tok, tokenizer
                )
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

            if frontier not in frontier_history:
                frontier_history[frontier] = []
            frontier_history[frontier].append({
                "step": i, "t": timesteps[i].item(), **metrics,
            })

            # Core predictor step (uses standard discrete x)
            x = predictor.update_fn(score_fn, x, t, dt)

            # Re-enforce LTR
            if frontier < 1024:
                x[:, frontier + 1:] = MASK

                if x[0, frontier].item() != MASK:
                    committed_token = x[0, frontier].item()
                    committed_str = tokenizer.decode([committed_token])
                    steps_taken = i - current_frontier_start_step + 1

                    correct = (committed_token == target_tok) if target_tok is not None else None

                    last_hist = frontier_history.get(frontier, [{}])[-1]

                    entry = {
                        "position": frontier,
                        "step": i,
                        "steps_taken": steps_taken,
                        "committed_token_id": committed_token,
                        "committed_token": committed_str,
                        "final_surprisal": last_hist.get("surprisal"),
                        "final_entropy": last_hist.get("entropy"),
                        "cumulative_kl": cumulative_kl,
                    }
                    if target_tok is not None:
                        entry["target_token_id"] = target_tok
                        entry["target_token"] = tokenizer.decode([target_tok])
                        entry["correct"] = correct

                    commitment_log.append(entry)

                    # Store the probability distribution at commitment for soft context
                    context_probs[frontier] = current_probs.clone()

                    # Override with target token for enforce-prefix
                    if target_tok is not None:
                        x[:, frontier] = target_tok

                    frontier += 1
                    prev_probs = None
                    cumulative_kl = 0.0
                    current_frontier_start_step = i + 1

            if i < 5 or i % max(1, steps // 20) == 0 or i == steps - 1:
                n_unmasked = (x[0] != MASK).sum().item()
                print(f"  step {i:5d} | frontier={frontier:4d} | unmasked={n_unmasked:5d}")

    # --- Export ---
    print(f"\n{'='*70}")
    print(f"Results: soft-context enforce-prefix LTR")
    print(f"{'='*70}\n")

    if commitment_log:
        n_show = min(30, len(commitment_log))
        print(f"Commitment log (first {n_show})")
        for entry in commitment_log[:n_show]:
            surp = entry.get("final_surprisal")
            surp_str = f"{surp:.2f}" if surp is not None else "N/A"
            print(f"  pos={entry['position']:3d} steps={entry['steps_taken']:4d} "
                  f"surp={surp_str:>8s} "
                  f"token={repr(entry['committed_token']):>15s} "
                  f"target={repr(entry.get('target_token','')):>15s} "
                  f"correct={entry.get('correct','')}")

    if output_path:
        serializable_history = {}
        for pos, hist_list in frontier_history.items():
            serializable_history[str(pos)] = [
                {k: v for k, v in h.items() if k != "probs"} for h in hist_list
            ]
            for h in serializable_history[str(pos)]:
                if h.get("top_k"):
                    h["top_k"] = [{"id": t[0], "token": t[1], "prob": t[2]} for t in h["top_k"]]

        output = {
            "config": {
                "model_path": model_path,
                "steps": steps,
                "seed": seed,
                "experiment_type": "soft_context",
                "enforce_prefix": True,
            },
            "tokenization": {
                "full_ids": full_ids,
                "sentence": sentence,
                "sentence_length": sentence_end,
            },
            "commitment_log": commitment_log,
            "frontier_history": serializable_history,
        }

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")

    return commitment_log, frontier_history


def main():
    parser = argparse.ArgumentParser(description="SEDD Soft-Context LTR")
    parser.add_argument("--model_path", type=str, default="louaaron/sedd-medium")
    parser.add_argument("--sentence", type=str, required=True)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    run_soft_context_experiment(
        model_path=args.model_path,
        sentence=args.sentence,
        steps=args.steps,
        device=device,
        seed=args.seed,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
