"""
Stage 2: SEDD Denoising Trajectory with Full Visibility

Runs the standard pc_sampler (unmodified) but logs every step to show
which tokens unmask, in what order, and how the sequence evolves.
Does NOT modify the model or sampling algorithm.

Usage:
    python sedd_stage2_trajectory.py --model_path louaaron/sedd-medium --steps 64
"""

import argparse
import torch

from sedd_helpers import load_sedd_model, get_sampling_score_fn
from sampling import get_predictor, Denoiser


def run_trajectory(model_path, steps, device, seed=42):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    print(f"{'='*70}")
    print(f"SEDD Trajectory Tracking")
    print(f"  model:  {model_path}")
    print(f"  steps:  {steps}")
    print(f"  device: {device}")
    print(f"  seed:   {seed}")
    print(f"{'='*70}\n")

    model, graph, noise, tokenizer = load_sedd_model(model_path, device)
    MASK = graph.dim - 1
    eps = 1e-5

    score_fn = get_sampling_score_fn(model)
    predictor = get_predictor("analytic")(graph, noise)
    denoiser = Denoiser(graph, noise)

    batch_dims = (1, 1024)
    x = graph.sample_limit(*batch_dims).to(device)
    timesteps = torch.linspace(1, eps, steps + 1, device=device)
    dt = (1 - eps) / steps

    # Tracking state
    prev_x = x.clone()
    unmasked_counts = []
    change_counts = []
    step_log = []

    print(f"Initial state: all {(x == MASK).sum().item()} positions are MASK ({MASK})\n")
    print(f"{'Step':>6} {'t':>8} {'sigma':>10} {'unmasked':>10} {'new_unmask':>12} {'changed':>10}")
    print("-" * 70)

    with torch.no_grad():
        for i in range(steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            sigma_val = noise(t)[0]

            x = predictor.update_fn(score_fn, x, t, dt)

            n_unmasked = (x[0] != MASK).sum().item()
            n_changed = (x[0] != prev_x[0]).sum().item()

            newly_unmasked = ((prev_x[0] == MASK) & (x[0] != MASK)).sum().item()

            unmasked_counts.append(n_unmasked)
            change_counts.append(n_changed)
            step_log.append({
                "step": i,
                "t": timesteps[i].item(),
                "sigma": sigma_val[0, 0].item(),
                "n_unmasked": n_unmasked,
                "newly_unmasked": newly_unmasked,
                "n_changed": n_changed,
            })

            if i < 10 or i % max(1, steps // 20) == 0 or i == steps - 1:
                print(
                    f"{i:6d} {timesteps[i].item():8.4f} "
                    f"{sigma_val[0,0].item():10.4f} "
                    f"{n_unmasked:10d} {newly_unmasked:12d} {n_changed:10d}"
                )

            prev_x = x.clone()

        # Final denoising step
        print(f"\n{'='*70}")
        print("Final denoising step")
        print(f"{'='*70}\n")

        t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
        x = denoiser.update_fn(score_fn, x, t)

        n_unmasked_final = (x[0] != MASK).sum().item()
        n_still_masked = (x[0] == MASK).sum().item()
        print(f"  Unmasked after denoiser: {n_unmasked_final} / 1024")
        print(f"  Still masked:            {n_still_masked}")
        print()

    # --- Verification ---
    print(f"{'='*70}")
    print("Verification")
    print(f"{'='*70}\n")

    monotonic = all(
        unmasked_counts[i] >= unmasked_counts[i - 1]
        for i in range(1, len(unmasked_counts))
    )
    print(f"  Unmasked count monotonically increasing? {monotonic}")
    if not monotonic:
        decreases = sum(
            1 for i in range(1, len(unmasked_counts))
            if unmasked_counts[i] < unmasked_counts[i - 1]
        )
        print(f"  (Note: {decreases} decreases found -- this can happen since")
        print(f"   unmasked tokens cannot change but MASK can re-sample to MASK)")

    all_unmasked = n_still_masked == 0
    print(f"  All positions unmasked after denoiser?   {all_unmasked}")
    print()

    # --- Unmasking order analysis ---
    print(f"{'='*70}")
    print("Unmasking order analysis (first 20 positions to unmask)")
    print(f"{'='*70}\n")

    # Re-run to track per-position unmasking order
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    x2 = graph.sample_limit(*batch_dims).to(device)
    first_unmask_step = torch.full((1024,), -1, dtype=torch.long)

    with torch.no_grad():
        for i in range(steps):
            t = timesteps[i] * torch.ones(x2.shape[0], 1, device=device)
            prev = x2.clone()
            x2 = predictor.update_fn(score_fn, x2, t, dt)

            newly = (prev[0] == MASK) & (x2[0] != MASK)
            newly_positions = newly.nonzero(as_tuple=True)[0]
            for pos in newly_positions:
                if first_unmask_step[pos] == -1:
                    first_unmask_step[pos] = i

        t = timesteps[-1] * torch.ones(x2.shape[0], 1, device=device)
        x2 = denoiser.update_fn(score_fn, x2, t)
        still_mask = (first_unmask_step == -1)
        first_unmask_step[still_mask] = steps

    order = first_unmask_step.argsort()
    print(f"{'Position':>10} {'Unmask Step':>12} {'Token':>30}")
    print("-" * 55)
    for rank in range(min(20, 1024)):
        pos = order[rank].item()
        step_val = first_unmask_step[pos].item()
        tok = x2[0, pos].item()
        tok_str = tokenizer.decode([tok])
        print(f"{pos:10d} {step_val:12d} {repr(tok_str):>30s}")

    print()

    # --- Decoded output ---
    print(f"{'='*70}")
    print("Decoded output (first 200 chars)")
    print(f"{'='*70}\n")

    text = tokenizer.decode(x[0].tolist())
    print(text[:200])
    print("...")
    print()

    # --- Unmasking curve summary ---
    print(f"{'='*70}")
    print("Unmasking curve summary")
    print(f"{'='*70}\n")

    quartiles = [0, len(unmasked_counts) // 4, len(unmasked_counts) // 2,
                 3 * len(unmasked_counts) // 4, len(unmasked_counts) - 1]
    for q in quartiles:
        print(f"  Step {q:5d}: {unmasked_counts[q]:5d} / 1024 unmasked")
    print()
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="SEDD Stage 2: Trajectory Tracking")
    parser.add_argument("--model_path", type=str, default="louaaron/sedd-medium")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    run_trajectory(args.model_path, args.steps, device, seed=args.seed)


if __name__ == "__main__":
    main()
