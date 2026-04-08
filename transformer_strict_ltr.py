"""
SEDD Ablation Experiments: Strict LTR, Causal Attention, Distance-Based Forgetting

Unified sampling loop with composable flags:
  --ltr              Strict left-to-right (overwrite future positions to MASK)
  --causal           Causal attention (pretrained model uses non-causal)
  --renoise          Distance-based forgetting (requires --ltr)
  --enforce_prefix   Override committed token with actual target at each position;
                     implies --ltr. Stops tracking after sentence-length tokens committed.
  --sentence         Full sentence for enforce-prefix mode (tokenized with <|endoftext|> prefix)
  --output_path      Path to write JSON results (commitment_log + frontier_history)

Tracks per-step frontier metrics: surprisal, entropy, KL divergence, top-K.

Usage:
  # Enforce-prefix LTR on a sentence (primary mode for experiments)
  python transformer_strict_ltr.py --enforce_prefix \\
      --sentence "The horse raced past the barn fell ." \\
      --output_path LTR_SAP/test_output.json

  # Legacy: Standard parallel, non-causal (baseline)
  python transformer_strict_ltr.py --prefix "The horse raced past the barn"

  # Legacy: Strict LTR
  python transformer_strict_ltr.py --prefix "The horse raced past the barn" --ltr
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
    renoise_fn,
    compute_frontier_metrics,
    compute_kl_divergence,
    tokenize_sentence,
    log_step_state,
)
from sampling import get_predictor, Denoiser


def run_experiment(
    model_path,
    prefix,
    target,
    sentence,
    steps,
    device,
    ltr,
    causal,
    renoise,
    renoise_sigma,
    enforce_prefix,
    batch_size,
    seed,
    output_path,
    model_bundle=None,
):
    """Run a single SEDD denoising experiment.

    Args:
        model_bundle: optional (model, graph, noise, tokenizer) tuple to avoid
                      reloading the model when called from a batch runner.
    """
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    if enforce_prefix:
        ltr = True

    config_desc = []
    if enforce_prefix:
        config_desc.append("enforce-prefix-LTR")
    elif ltr:
        config_desc.append("LTR")
    else:
        config_desc.append("parallel")
    config_desc.append("causal" if causal else "noncausal")
    if renoise:
        config_desc.append(f"renoise(sigma={renoise_sigma})")
    config_str = " | ".join(config_desc)

    print(f"{'='*70}")
    print(f"SEDD Experiment: {config_str}")
    print(f"{'='*70}")

    if renoise and not ltr:
        raise ValueError("--renoise requires --ltr (re-noising needs a frontier)")

    if model_bundle is not None:
        model, graph, noise, tokenizer = model_bundle
    else:
        model, graph, noise, tokenizer = load_sedd_model(model_path, device)

    MASK = graph.dim - 1
    eps = 1e-5

    if causal:
        set_causal_mode(model, causal=True)
    else:
        set_causal_mode(model, causal=False)

    # --- Tokenization ---
    if enforce_prefix:
        if not sentence:
            raise ValueError("--enforce_prefix requires --sentence")
        full_ids = tokenize_sentence(tokenizer, sentence)
        prefix_len = 1  # just <|endoftext|>
        target_ids = full_ids
        sentence_end = len(full_ids)
        print(f"  sentence:      {repr(sentence)}")
        print(f"  full_ids ({len(full_ids)}): {full_ids[:20]}{'...' if len(full_ids)>20 else ''}")
        print(f"  decoded:       {repr(tokenizer.decode(full_ids))}")
        prefix_tensor = torch.tensor([full_ids[0]], device=device)[None].repeat(batch_size, 1)
    else:
        prefix_ids = tokenizer(prefix).input_ids
        prefix_len = len(prefix_ids)
        prefix_tensor = torch.tensor(prefix_ids, device=device)[None].repeat(batch_size, 1)
        target_ids = tokenizer(target).input_ids if target else None
        sentence_end = None
        print(f"  prefix ({prefix_len}): {repr(tokenizer.decode(prefix_ids))}")
        if target_ids:
            print(f"  target ({len(target_ids)}): {repr(tokenizer.decode(target_ids))}")
    print()

    score_fn = get_sampling_score_fn(model)
    predictor = get_predictor("analytic")(graph, noise)
    denoiser = Denoiser(graph, noise)

    x = graph.sample_limit(batch_size, 1024).to(device)
    timesteps = torch.linspace(1, eps, steps + 1, device=device)
    dt = (1 - eps) / steps

    frontier = prefix_len if ltr else None

    frontier_history = {}
    commitment_log = []
    argmax_history = []
    current_frontier_start_step = 0
    prev_probs = None
    cumulative_kl = 0.0

    print(f"Sampling loop ({steps} steps)...\n")

    with torch.no_grad():
        for i in range(steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)

            # Fix prefix
            if enforce_prefix:
                committed_so_far = min(frontier, sentence_end)
                x[:, :committed_so_far] = torch.tensor(
                    full_ids[:committed_so_far], device=device
                )[None]
            else:
                x[:, :prefix_len] = prefix_tensor

            # LTR enforcement
            if ltr and frontier < 1024:
                x[:, frontier + 1:] = MASK

            # Re-noising
            if renoise and ltr:
                x = renoise_fn(x, graph, frontier, prefix_len, renoise_sigma)

            # Early stop for enforce-prefix
            if enforce_prefix and frontier >= sentence_end:
                break

            # Collect pre-step score at frontier
            if ltr and frontier < 1024:
                curr_sigma = noise(t)[0]
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    raw_score = score_fn(x, curr_sigma)

                target_tok = target_ids[frontier] if (target_ids and frontier < len(target_ids)) else None
                if target_tok is not None:
                    metrics = compute_frontier_metrics(
                        raw_score, frontier, target_tok, tokenizer
                    )
                else:
                    metrics = compute_frontier_metrics(
                        raw_score, frontier, 0, tokenizer
                    )
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
                argmax_history.append(metrics["argmax_id"])

                if frontier not in frontier_history:
                    frontier_history[frontier] = []
                frontier_history[frontier].append({
                    "step": i,
                    "t": timesteps[i].item(),
                    **metrics,
                })

            # Core predictor step
            x = predictor.update_fn(score_fn, x, t, dt)

            # Re-enforce LTR
            if ltr and frontier < 1024:
                x[:, frontier + 1:] = MASK

                if x[0, frontier].item() != MASK:
                    committed_token = x[0, frontier].item()
                    committed_str = tokenizer.decode([committed_token])
                    steps_taken = i - current_frontier_start_step + 1

                    target_tok = target_ids[frontier] if (target_ids and frontier < len(target_ids)) else None
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
                    if enforce_prefix and target_tok is not None:
                        entry["target_token_id"] = target_tok
                        entry["target_token"] = tokenizer.decode([target_tok])
                        entry["correct"] = correct

                    commitment_log.append(entry)

                    # Enforce-prefix: override with target token
                    if enforce_prefix and target_tok is not None:
                        x[:, frontier] = target_tok

                    frontier += 1
                    argmax_history = []
                    prev_probs = None
                    cumulative_kl = 0.0
                    current_frontier_start_step = i + 1

            # Periodic logging
            if i < 5 or i % max(1, steps // 20) == 0 or i == steps - 1:
                n_unmasked = (x[0] != MASK).sum().item()
                if ltr:
                    state_str = log_step_state(x, frontier, prefix_len, MASK, tokenizer)
                    print(f"  step {i:5d} | frontier={frontier:4d} | unmasked={n_unmasked:5d} | {state_str}")
                else:
                    print(f"  step {i:5d} | unmasked={n_unmasked:5d}")

        # Final denoising (skip if enforce-prefix already done)
        if not (enforce_prefix and frontier >= sentence_end):
            print(f"\nFinal denoising step...")
            if ltr:
                if enforce_prefix:
                    committed_so_far = min(frontier, sentence_end) if sentence_end else frontier
                    x[:, :committed_so_far] = torch.tensor(
                        full_ids[:committed_so_far], device=device
                    )[None]
                else:
                    x[:, :prefix_len] = prefix_tensor

                if frontier < 1024:
                    x[:, frontier + 1:] = MASK
                if renoise:
                    x = renoise_fn(x, graph, frontier, prefix_len, renoise_sigma)

            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
            x = denoiser.update_fn(score_fn, x, t)

            if ltr:
                if enforce_prefix:
                    committed_so_far = min(frontier, sentence_end) if sentence_end else frontier
                    x[:, :committed_so_far] = torch.tensor(
                        full_ids[:committed_so_far], device=device
                    )[None]
                else:
                    x[:, :prefix_len] = prefix_tensor

                while frontier < 1024 and x[0, frontier].item() != MASK:
                    if enforce_prefix and frontier >= sentence_end:
                        break
                    committed_token = x[0, frontier].item()
                    committed_str = tokenizer.decode([committed_token])
                    target_tok = target_ids[frontier] if (target_ids and frontier < len(target_ids)) else None

                    entry = {
                        "position": frontier,
                        "step": steps,
                        "steps_taken": steps - current_frontier_start_step + 1,
                        "committed_token_id": committed_token,
                        "committed_token": committed_str,
                        "final_surprisal": None,
                        "final_entropy": None,
                        "cumulative_kl": 0.0,
                    }
                    if enforce_prefix and target_tok is not None:
                        entry["target_token_id"] = target_tok
                        entry["target_token"] = tokenizer.decode([target_tok])
                        entry["correct"] = (committed_token == target_tok)
                        x[:, frontier] = target_tok

                    commitment_log.append(entry)
                    frontier += 1
                    current_frontier_start_step = steps

    # --- Print results ---
    print(f"\n{'='*70}")
    print(f"Results: {config_str}")
    print(f"{'='*70}\n")

    if ltr and commitment_log:
        n_show = min(30, len(commitment_log))
        print(f"Commitment log (first {n_show} positions)")
        print(f"{'Pos':>5} {'Step':>6} {'#Steps':>7} {'Surp':>8} {'Entropy':>8} {'CumKL':>8} {'Token':>20} {'Target':>20} {'OK':>4}")
        print("-" * 95)
        for entry in commitment_log[:n_show]:
            surp_str = f"{entry['final_surprisal']:.2f}" if entry['final_surprisal'] is not None else "N/A"
            ent_str = f"{entry['final_entropy']:.2f}" if entry['final_entropy'] is not None else "N/A"
            target_str = repr(entry.get('target_token', '')) if enforce_prefix else ""
            correct_str = str(entry.get('correct', ''))
            print(
                f"{entry['position']:5d} {entry['step']:6d} "
                f"{entry['steps_taken']:7d} {surp_str:>8s} {ent_str:>8s} "
                f"{entry['cumulative_kl']:8.2f} "
                f"{repr(entry['committed_token']):>20s} {target_str:>20s} {correct_str:>4s}"
            )
        print()

        if enforce_prefix:
            n_correct = sum(1 for e in commitment_log if e.get('correct'))
            print(f"  Accuracy: {n_correct}/{len(commitment_log)} ({n_correct/len(commitment_log)*100:.1f}%)")
        avg_steps = np.mean([e["steps_taken"] for e in commitment_log])
        print(f"  Avg steps/token: {avg_steps:.1f}")
        print()

    # --- Export to JSON ---
    if output_path:
        serializable_history = {}
        for pos, hist_list in frontier_history.items():
            serializable_history[str(pos)] = [
                {k: v for k, v in h.items() if k != "probs"} for h in hist_list
            ]
            for h in serializable_history[str(pos)]:
                if h.get("top_k"):
                    h["top_k"] = [
                        {"id": t[0], "token": t[1], "prob": t[2]} for t in h["top_k"]
                    ]

        output = {
            "config": {
                "model_path": model_path,
                "steps": steps,
                "seed": seed,
                "enforce_prefix": enforce_prefix,
                "ltr": ltr,
                "causal": causal,
                "renoise": renoise,
                "renoise_sigma": renoise_sigma,
            },
            "tokenization": {
                "full_ids": full_ids if enforce_prefix else (target_ids or []),
                "sentence": sentence if enforce_prefix else (target or prefix),
                "sentence_length": sentence_end if enforce_prefix else None,
            },
            "commitment_log": commitment_log,
            "frontier_history": serializable_history,
        }

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"Results saved to {output_path}")

    return commitment_log, frontier_history


def main():
    parser = argparse.ArgumentParser(
        description="SEDD Ablation: LTR / Causal / Re-noising / Enforce-prefix experiments"
    )
    parser.add_argument("--model_path", type=str, default="louaaron/sedd-medium")
    parser.add_argument("--prefix", type=str, default="The horse raced past the barn")
    parser.add_argument(
        "--target", type=str, default=None,
        help="Full target sentence for surprisal (legacy mode)",
    )
    parser.add_argument(
        "--sentence", type=str, default=None,
        help="Full sentence for enforce-prefix mode (tokenized with <|endoftext|> prefix)",
    )
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Path to write JSON results",
    )

    parser.add_argument("--ltr", action="store_true", help="Strict left-to-right generation")
    parser.add_argument("--causal", action="store_true", help="Use causal attention")
    parser.add_argument("--renoise", action="store_true", help="Enable distance-based forgetting")
    parser.add_argument("--renoise_sigma", type=float, default=1.0, help="Base sigma for re-noising")
    parser.add_argument("--enforce_prefix", action="store_true",
                        help="Override committed token with actual target (implies --ltr)")

    args = parser.parse_args()
    device = torch.device(args.device)

    run_experiment(
        model_path=args.model_path,
        prefix=args.prefix,
        target=args.target,
        sentence=args.sentence,
        steps=args.steps,
        device=device,
        ltr=args.ltr,
        causal=args.causal,
        renoise=args.renoise,
        renoise_sigma=args.renoise_sigma,
        enforce_prefix=args.enforce_prefix,
        batch_size=args.batch_size,
        seed=args.seed,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
