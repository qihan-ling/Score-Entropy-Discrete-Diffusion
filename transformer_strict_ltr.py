"""
SEDD Ablation Experiments: Strict LTR, Causal Attention, Distance-Based Forgetting

Unified sampling loop with three composable flags:
  --ltr       Strict left-to-right (overwrite future positions to MASK)
  --causal    Causal attention (pretrained model uses non-causal)
  --renoise   Distance-based forgetting (requires --ltr)

Tracks per-step frontier metrics: surprisal, entropy, top-K, flickering.

Usage:
  # Standard parallel, non-causal, no re-noising (baseline)
  python transformer_strict_ltr.py --prefix "The horse raced past the barn"

  # Strict LTR
  python transformer_strict_ltr.py --prefix "The horse raced past the barn" --ltr

  # Strict LTR + causal
  python transformer_strict_ltr.py --prefix "The horse raced past the barn" --ltr --causal

  # Strict LTR + re-noising
  python transformer_strict_ltr.py --prefix "The horse raced past the barn" --ltr --renoise --renoise_sigma 1.0

  # Full ablation
  python transformer_strict_ltr.py --prefix "The horse raced past the barn" --ltr --causal --renoise --renoise_sigma 1.0
"""

import argparse
import torch
import numpy as np

from sedd_helpers import (
    load_sedd_model,
    get_sampling_score_fn,
    set_causal_mode,
    renoise_fn,
    compute_frontier_metrics,
    compute_flickering,
    log_step_state,
)
from sampling import get_predictor, Denoiser


def run_experiment(
    model_path,
    prefix,
    target,
    steps,
    device,
    ltr,
    causal,
    renoise,
    renoise_sigma,
    batch_size,
    seed,
):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    # --- Setup ---
    config_desc = []
    if ltr:
        config_desc.append("LTR")
    else:
        config_desc.append("parallel")
    config_desc.append("causal" if causal else "noncausal")
    if renoise:
        config_desc.append(f"renoise(sigma={renoise_sigma})")
    else:
        config_desc.append("no-renoise")
    config_str = " | ".join(config_desc)

    print(f"{'='*70}")
    print(f"SEDD Experiment: {config_str}")
    print(f"{'='*70}")
    print(f"  model:    {model_path}")
    print(f"  prefix:   {repr(prefix)}")
    print(f"  target:   {repr(target)}")
    print(f"  steps:    {steps}")
    print(f"  seed:     {seed}")
    print(f"  ltr:      {ltr}")
    print(f"  causal:   {causal}")
    print(f"  renoise:  {renoise} (sigma={renoise_sigma})")
    print(f"{'='*70}\n")

    if renoise and not ltr:
        raise ValueError("--renoise requires --ltr (re-noising needs a frontier)")

    model, graph, noise, tokenizer = load_sedd_model(model_path, device)
    MASK = graph.dim - 1
    eps = 1e-5

    # Apply causal mode
    if causal:
        print("Setting causal attention mode...")
        set_causal_mode(model, causal=True)
    else:
        print("Setting non-causal attention mode (matches pretrained weights)...")
        set_causal_mode(model, causal=False)

    # Tokenize prefix and target
    prefix_ids = tokenizer(prefix).input_ids
    prefix_len = len(prefix_ids)
    prefix_tensor = torch.tensor(prefix_ids, device=device)[None].repeat(batch_size, 1)

    target_ids = None
    if target:
        target_ids = tokenizer(target).input_ids
        print(f"  prefix tokens ({prefix_len}): {prefix_ids}")
        print(f"  prefix decoded: {repr(tokenizer.decode(prefix_ids))}")
        print(f"  target tokens ({len(target_ids)}): {target_ids}")
        print(f"  target decoded: {repr(tokenizer.decode(target_ids))}")
    else:
        print(f"  prefix tokens ({prefix_len}): {prefix_ids}")
        print(f"  prefix decoded: {repr(tokenizer.decode(prefix_ids))}")
    print()

    # --- Scoring function ---
    score_fn = get_sampling_score_fn(model)

    predictor = get_predictor("analytic")(graph, noise)
    denoiser = Denoiser(graph, noise)

    # --- Initialize ---
    x = graph.sample_limit(batch_size, 1024).to(device)
    timesteps = torch.linspace(1, eps, steps + 1, device=device)
    dt = (1 - eps) / steps

    frontier = prefix_len if ltr else None

    # Per-frontier-position tracking (only meaningful with --ltr)
    frontier_history = {}  # pos -> list of per-step dicts
    commitment_log = []    # ordered list of (pos, step, token_id, token_str)
    argmax_history = []    # list of argmax at frontier per step (for flickering)
    current_frontier_start_step = 0

    print(f"{'='*70}")
    print(f"Sampling loop ({steps} steps)")
    print(f"{'='*70}\n")

    with torch.no_grad():
        for i in range(steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)

            # --- Layer 1: Fix prefix (always) ---
            x[:, :prefix_len] = prefix_tensor

            # --- Layer 2: LTR enforcement ---
            if ltr:
                x[:, frontier + 1:] = MASK

            # --- Layer 3: Re-noising ---
            if renoise and ltr:
                x = renoise_fn(x, graph, frontier, prefix_len, renoise_sigma)

            # --- Collect pre-step score at frontier (for metrics) ---
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

                argmax_history.append(metrics["argmax_id"])

                if frontier not in frontier_history:
                    frontier_history[frontier] = []
                frontier_history[frontier].append({
                    "step": i,
                    "t": timesteps[i].item(),
                    **metrics,
                })

            # --- Core: predictor step ---
            x = predictor.update_fn(score_fn, x, t, dt)

            # --- Layer 2b: Re-enforce LTR ---
            if ltr:
                x[:, frontier + 1:] = MASK

                # Advance frontier
                if frontier < 1024 and x[0, frontier].item() != MASK:
                    committed_token = x[0, frontier].item()
                    committed_str = tokenizer.decode([committed_token])
                    steps_taken = i - current_frontier_start_step + 1
                    flicker = compute_flickering(argmax_history)

                    commitment_log.append({
                        "position": frontier,
                        "step": i,
                        "steps_taken": steps_taken,
                        "token_id": committed_token,
                        "token": committed_str,
                        "flickering": flicker,
                    })

                    frontier += 1
                    argmax_history = []
                    current_frontier_start_step = i + 1

            # --- Periodic logging ---
            if i < 5 or i % max(1, steps // 20) == 0 or i == steps - 1:
                n_unmasked = (x[0] != MASK).sum().item()
                if ltr:
                    state_str = log_step_state(x, frontier, prefix_len, MASK, tokenizer)
                    print(f"  step {i:5d} | frontier={frontier:4d} | unmasked={n_unmasked:5d} | {state_str}")
                else:
                    print(f"  step {i:5d} | unmasked={n_unmasked:5d}")

        # --- Final denoising ---
        print(f"\nFinal denoising step...")
        if ltr:
            x[:, :prefix_len] = prefix_tensor
            if frontier < 1024:
                x[:, frontier + 1:] = MASK
            if renoise:
                x = renoise_fn(x, graph, frontier, prefix_len, renoise_sigma)

        t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
        x_before_denoise = x.clone()
        x = denoiser.update_fn(score_fn, x, t)

        if ltr:
            # Fix prefix in final output
            x[:, :prefix_len] = prefix_tensor
            # Positions after frontier that denoiser may have filled
            while frontier < 1024 and x[0, frontier].item() != MASK:
                committed_token = x[0, frontier].item()
                committed_str = tokenizer.decode([committed_token])
                commitment_log.append({
                    "position": frontier,
                    "step": steps,
                    "steps_taken": steps - current_frontier_start_step + 1,
                    "token_id": committed_token,
                    "token": committed_str,
                    "flickering": compute_flickering(argmax_history),
                })
                frontier += 1
                argmax_history = []
                current_frontier_start_step = steps

    # --- Results ---
    print(f"\n{'='*70}")
    print(f"Results: {config_str}")
    print(f"{'='*70}\n")

    # Decoded output
    text = tokenizer.decode(x[0].tolist())
    print(f"Generated text (first 300 chars):")
    print(f"  {text[:300]}")
    print()

    # Commitment log (LTR only)
    if ltr and commitment_log:
        print(f"{'='*70}")
        print(f"Commitment log (first {min(30, len(commitment_log))} positions)")
        print(f"{'='*70}\n")

        print(f"{'Pos':>5} {'Step':>6} {'#Steps':>7} {'Flicker':>8} {'Token':>30}")
        print("-" * 60)
        for entry in commitment_log[:30]:
            print(
                f"{entry['position']:5d} {entry['step']:6d} "
                f"{entry['steps_taken']:7d} {entry['flickering']:8d} "
                f"{repr(entry['token']):>30s}"
            )
        print()

        total_committed = len(commitment_log)
        avg_steps = np.mean([e["steps_taken"] for e in commitment_log])
        avg_flicker = np.mean([e["flickering"] for e in commitment_log])
        print(f"  Total committed: {total_committed}")
        print(f"  Avg steps/token: {avg_steps:.1f}")
        print(f"  Avg flickering:  {avg_flicker:.1f}")
        print()

    # Frontier metrics (LTR only)
    if ltr and frontier_history:
        print(f"{'='*70}")
        print(f"Frontier metrics (per-position summary)")
        print(f"{'='*70}\n")

        print(f"{'Pos':>5} {'Token':>20} {'Avg Surp':>10} {'Avg Entropy':>12} {'#Steps':>8}")
        print("-" * 60)

        for pos in sorted(frontier_history.keys()):
            hist = frontier_history[pos]
            if not hist:
                continue

            surprisals = [h["surprisal"] for h in hist if h["surprisal"] is not None]
            entropies = [h["entropy"] for h in hist]

            avg_surp = np.mean(surprisals) if surprisals else float("nan")
            avg_ent = np.mean(entropies) if entropies else float("nan")

            committed = [e for e in commitment_log if e["position"] == pos]
            tok_str = committed[0]["token"] if committed else "?"

            print(
                f"{pos:5d} {repr(tok_str):>20s} {avg_surp:10.3f} "
                f"{avg_ent:12.3f} {len(hist):8d}"
            )

            # Show top-K from the last logged step before commitment
            if hist:
                last = hist[-1]
                top_k_str = ", ".join(
                    f"{repr(t[1])}:{t[2]:.3f}" for t in last["top_k"][:5]
                )
                print(f"      last step top-5: {top_k_str}")
        print()

    # Summary
    n_remaining_mask = (x[0] == MASK).sum().item()
    print(f"Final: {1024 - n_remaining_mask} / 1024 unmasked, {n_remaining_mask} still MASK")
    print(f"\nDone. Config: {config_str}")


def main():
    parser = argparse.ArgumentParser(
        description="SEDD Ablation: LTR / Causal / Re-noising experiments"
    )
    parser.add_argument("--model_path", type=str, default="louaaron/sedd-medium")
    parser.add_argument("--prefix", type=str, default="The horse raced past the barn")
    parser.add_argument(
        "--target", type=str, default="The horse raced past the barn fell",
        help="Full target sentence for surprisal computation",
    )
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Ablation flags
    parser.add_argument("--ltr", action="store_true", help="Strict left-to-right generation")
    parser.add_argument("--causal", action="store_true", help="Use causal attention")
    parser.add_argument("--renoise", action="store_true", help="Enable distance-based forgetting")
    parser.add_argument("--renoise_sigma", type=float, default=1.0, help="Base sigma for re-noising")

    args = parser.parse_args()
    device = torch.device(args.device)

    run_experiment(
        model_path=args.model_path,
        prefix=args.prefix,
        target=args.target,
        steps=args.steps,
        device=device,
        ltr=args.ltr,
        causal=args.causal,
        renoise=args.renoise,
        renoise_sigma=args.renoise_sigma,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
