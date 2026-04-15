"""
Compare five context modes on the same sentences.

Runs soft-context and Monte Carlo scripts on a full subset, loads existing
strict-LTR and strict-LTR+renoise results, and includes an all-masked helper.

Modes:
  1. hard     - enforce-prefix strict-LTR (baseline)
  2. soft     - soft probability embeddings at prior positions
  3. montecarlo - N sampled discrete contexts, averaged scores
  4. renoise  - distance-based forgetting
  5. masked   - all prior positions MASK

Produces:
  - CSV table: per-word steps by mode
  - Pairwise scatter plots
  - Bar chart with 95% CIs
  - Wilcoxon signed-rank tests
  - Gap-recovery metric

Usage:
  python LTR_SAP/analysis/compare_context_modes.py \
      --model_path louaaron/sedd-medium \
      --subset Agreement \
      --output_dir LTR_SAP/analysis/results/context_comparison
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import json
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

import torch

from utils import (
    get_sap_files, get_subset_name, iter_sap_items,
    load_all_outputs, extract_word_metrics,
    setup_matplotlib, condition_palette,
)


def run_masked_baseline(sentence, steps, device, model_bundle):
    """Run enforce-prefix LTR with all prior positions masked (no context)."""
    from transformer_strict_ltr import run_experiment

    model, graph, noise, tokenizer = model_bundle
    MASK = graph.dim - 1

    # Use the standard strict-LTR but we'll manually set up a wrapper
    # Actually, the cleanest approach: run enforce-prefix but re-mask all
    # prior positions before each forward pass. We'll implement inline.

    from sedd_helpers import (
        get_sampling_score_fn, tokenize_sentence,
        compute_frontier_metrics, compute_kl_divergence,
    )
    from sampling import get_predictor, Denoiser

    full_ids = tokenize_sentence(tokenizer, sentence)
    prefix_len = 1
    target_ids = full_ids
    sentence_end = len(full_ids)
    eps = 1e-5

    score_fn = get_sampling_score_fn(model)
    predictor = get_predictor("analytic")(graph, noise)

    x = graph.sample_limit(1, 1024).to(device)
    timesteps = torch.linspace(1, eps, steps + 1, device=device)
    dt = (1 - eps) / steps

    frontier = prefix_len
    commitment_log = []
    current_frontier_start_step = 0

    with torch.no_grad():
        for i in range(steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)

            # Keep position 0 as <|endoftext|>, but mask everything else before frontier
            x[:, 0] = full_ids[0]
            for pos in range(1, frontier):
                x[:, pos] = MASK  # masked context

            if frontier < 1024:
                x[:, frontier + 1:] = MASK

            if frontier >= sentence_end:
                break

            x = predictor.update_fn(score_fn, x, t, dt)

            # Re-enforce
            x[:, 0] = full_ids[0]
            for pos in range(1, frontier):
                x[:, pos] = MASK

            if frontier < 1024:
                x[:, frontier + 1:] = MASK

                if x[0, frontier].item() != MASK:
                    committed_token = x[0, frontier].item()
                    steps_taken = i - current_frontier_start_step + 1
                    target_tok = target_ids[frontier] if frontier < len(target_ids) else None

                    commitment_log.append({
                        "position": frontier,
                        "step": i,
                        "steps_taken": steps_taken,
                        "committed_token_id": committed_token,
                        "committed_token": tokenizer.decode([committed_token]),
                    })

                    if target_tok is not None:
                        x[:, frontier] = target_tok

                    frontier += 1
                    current_frontier_start_step = i + 1

    return commitment_log


def collect_steps_from_log(commitment_log, sentence):
    """Extract word-position -> steps_taken from a commitment log."""
    words = sentence.split()
    # Simple mapping: token position = word position + 1 (for <|endoftext|>)
    # This is approximate; multi-token words may have summed steps
    result = {}
    for entry in commitment_log:
        pos = entry["position"]
        word_idx = pos - 1  # subtract <|endoftext|>
        if word_idx < 0 or word_idx >= len(words):
            continue
        if word_idx not in result:
            result[word_idx] = 0
        result[word_idx] += entry["steps_taken"]
    return result


def main():
    parser = argparse.ArgumentParser(description="Compare context modes")
    parser.add_argument("--model_path", type=str, default="louaaron/sedd-medium")
    parser.add_argument("--subset", type=str, default="Agreement")
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--mc_samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="LTR_SAP/analysis/results/context_comparison")
    parser.add_argument("--max_items", type=int, default=None, help="Limit items for testing")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model once
    from sedd_helpers import load_sedd_model
    print("Loading SEDD model...")
    model, graph, noise, tokenizer = load_sedd_model(args.model_path, device)
    model_bundle = (model, graph, noise, tokenizer)
    print("Model loaded.\n")

    from transformer_soft_context import run_soft_context_experiment
    from transformer_montecarlo import run_montecarlo_experiment
    from transformer_strict_ltr import run_experiment as run_strict_ltr

    # Collect items
    csv_files = get_sap_files()
    csv_path = None
    for f in csv_files:
        if get_subset_name(f) == args.subset:
            csv_path = f
            break
    if csv_path is None:
        print(f"Subset {args.subset} not found")
        return

    items = list(iter_sap_items(csv_path))
    if args.max_items:
        items = items[:args.max_items]

    all_rows = []

    for idx, item_info in enumerate(items):
        sentence = item_info["sentence"]
        item_id = item_info["item"]
        condition = item_info["condition"]
        words = sentence.split()

        print(f"\n{'='*60}")
        print(f"Item {idx+1}/{len(items)}: item={item_id}, cond={condition}")
        print(f"  {sentence}")
        print(f"{'='*60}\n")

        steps_by_mode = {}

        # 1. Hard context (strict-LTR enforce-prefix)
        print("  [1/5] Hard context (enforce-prefix)...")
        hard_log, _ = run_strict_ltr(
            model_path=args.model_path, prefix="", target=None,
            sentence=sentence, steps=args.steps, device=device,
            ltr=True, causal=False, renoise=False, renoise_sigma=1.0,
            enforce_prefix=True, batch_size=1, seed=args.seed,
            output_path=None, model_bundle=model_bundle,
        )
        steps_by_mode["hard"] = collect_steps_from_log(hard_log, sentence)

        # 2. Soft context
        print("  [2/5] Soft context...")
        soft_log, _ = run_soft_context_experiment(
            model_path=args.model_path, sentence=sentence,
            steps=args.steps, device=device, seed=args.seed,
            output_path=None, model_bundle=model_bundle,
        )
        steps_by_mode["soft"] = collect_steps_from_log(soft_log, sentence)

        # 3. Monte Carlo
        print("  [3/5] Monte Carlo context...")
        mc_log, _ = run_montecarlo_experiment(
            model_path=args.model_path, sentence=sentence,
            steps=args.steps, device=device,
            mc_samples=args.mc_samples, seed=args.seed,
            output_path=None, model_bundle=model_bundle,
        )
        steps_by_mode["montecarlo"] = collect_steps_from_log(mc_log, sentence)

        # 4. Renoise
        print("  [4/5] Renoise context...")
        renoise_log, _ = run_strict_ltr(
            model_path=args.model_path, prefix="", target=None,
            sentence=sentence, steps=args.steps, device=device,
            ltr=True, causal=False, renoise=True, renoise_sigma=1.0,
            enforce_prefix=True, batch_size=1, seed=args.seed,
            output_path=None, model_bundle=model_bundle,
        )
        steps_by_mode["renoise"] = collect_steps_from_log(renoise_log, sentence)

        # 5. Masked
        print("  [5/5] Masked context...")
        masked_log = run_masked_baseline(sentence, args.steps, device, model_bundle)
        steps_by_mode["masked"] = collect_steps_from_log(masked_log, sentence)

        # Merge into rows
        for wpos in range(len(words)):
            row = {
                "item": item_id,
                "condition": condition,
                "word_pos": wpos,
                "word": words[wpos],
            }
            for mode in ["hard", "soft", "montecarlo", "renoise", "masked"]:
                row[f"steps_{mode}"] = steps_by_mode[mode].get(wpos, np.nan)
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df.to_csv(output_dir / f"{args.subset}_context_comparison.csv", index=False)
    print(f"\nComparison table saved to {output_dir / f'{args.subset}_context_comparison.csv'}")

    # --- Analysis ---
    plt = setup_matplotlib()

    modes = ["hard", "soft", "montecarlo", "renoise", "masked"]
    step_cols = [f"steps_{m}" for m in modes]
    valid = df.dropna(subset=step_cols)

    if valid.empty:
        print("No valid rows for comparison")
        return

    # Bar chart of mean steps
    means = [valid[f"steps_{m}"].mean() for m in modes]
    sems = [valid[f"steps_{m}"].sem() for m in modes]

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(modes))
    bars = ax.bar(x_pos, means, yerr=[s * 1.96 for s in sems], capsize=5,
                  color=["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(modes)
    ax.set_ylabel("Mean steps to commit")
    ax.set_title(f"Context Mode Comparison: {args.subset}")
    plt.tight_layout()
    plt.savefig(output_dir / f"{args.subset}_bar_chart.png")
    plt.close()

    # Pairwise scatter plots
    pairs = [("soft", "masked"), ("montecarlo", "masked"),
             ("soft", "montecarlo"), ("renoise", "masked")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, (m1, m2) in zip(axes.flat, pairs):
        c1, c2 = f"steps_{m1}", f"steps_{m2}"
        clean = valid[[c1, c2]].dropna()
        ax.scatter(clean[c2], clean[c1], alpha=0.3, s=10)
        lim = max(clean[c1].max(), clean[c2].max()) * 1.1
        ax.plot([0, lim], [0, lim], "k--", alpha=0.3)
        ax.set_xlabel(f"steps_{m2}")
        ax.set_ylabel(f"steps_{m1}")
        ax.set_title(f"{m1} vs {m2}")
    plt.suptitle(f"Pairwise Scatter: {args.subset}")
    plt.tight_layout()
    plt.savefig(output_dir / f"{args.subset}_scatter.png")
    plt.close()

    # Wilcoxon tests
    print(f"\n{'='*70}")
    print(f"Statistical tests")
    print(f"{'='*70}")
    test_pairs = [
        ("soft", "masked"),
        ("montecarlo", "masked"),
        ("soft", "montecarlo"),
        ("renoise", "masked"),
        ("hard", "masked"),
    ]
    test_results = []
    for m1, m2 in test_pairs:
        c1, c2 = f"steps_{m1}", f"steps_{m2}"
        clean = valid[[c1, c2]].dropna()
        if len(clean) > 5:
            stat, p = stats.wilcoxon(clean[c1], clean[c2])
            diff = clean[c1].mean() - clean[c2].mean()
            print(f"  {m1} vs {m2}: diff={diff:.2f}, W={stat:.0f}, p={p:.6f}")
            test_results.append({
                "pair": f"{m1}_vs_{m2}",
                "mean_1": clean[c1].mean(), "mean_2": clean[c2].mean(),
                "diff": diff, "W": stat, "p_value": p,
            })
    if test_results:
        pd.DataFrame(test_results).to_csv(
            output_dir / f"{args.subset}_wilcoxon_tests.csv", index=False
        )

    # Gap-recovery metric
    print(f"\n  Gap-recovery: (masked - X) / (masked - hard)")
    for mode in ["soft", "montecarlo", "renoise"]:
        c_mode = f"steps_{mode}"
        clean = valid[["steps_hard", "steps_masked", c_mode]].dropna()
        gap = clean["steps_masked"] - clean["steps_hard"]
        recovery = (clean["steps_masked"] - clean[c_mode]) / gap.clip(lower=1)
        print(f"    {mode}: mean recovery = {recovery.mean():.3f} "
              f"(median = {recovery.median():.3f})")

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
