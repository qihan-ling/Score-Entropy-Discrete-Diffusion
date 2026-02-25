"""
Experiment 1: Varying Final Denoising Threshold (Sigma Sweep)

Goal: Evaluate the relationship between noisiness of context and the
prediction of the target token at the disambiguating position.

Method:
- For each stimulus sentence, run the SEDD model at different sigma values
  (from high noise to near-clean)
- At each sigma, extract:
  1. P(target_token) from the previous position's logits (autoregressive-style)
  2. Full probability distribution metrics (entropy, rank, confidence, etc.)
- This reveals how the model's prediction of the critical word changes as
  context becomes cleaner.

Key insight: Higher sigma = noisier context → how much noise can the model
tolerate before losing its prediction of the disambiguating word?
"""

import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from sedd_experiment_utils import (
    SEDDModelWrapper, StimulusLoader, compute_surprisal, create_output_dir,
    GPT2ModelWrapper, compute_gpt2_baseline, NATS_TO_BITS
)


# Sigma values to sweep: from high noise to near-clean
DEFAULT_SIGMAS = [
    5.0, 3.0, 2.0, 1.5, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3,
    0.2, 0.15, 0.1, 0.05, 0.01, 0.005, 0.001, 1e-4, 1e-5
]


def run_sigma_sweep(model, stimuli, sigmas=None):
    """
    For each stimulus, evaluate the model at each sigma value and record
    how the prediction of the target token changes.

    Returns a DataFrame with one row per (stimulus, sigma) combination.
    """
    if sigmas is None:
        sigmas = DEFAULT_SIGMAS

    results = []

    for stim in tqdm(stimuli, desc="Stimuli"):
        tokens = model.tokenize(stim['sentence'])
        target_token_id = stim['target_token_id']
        target_pos = stim['target_token_pos']
        context_pos = target_pos - 1

        if context_pos < 0:
            continue

        with torch.no_grad():
            for sigma_val in sigmas:
                sigma = torch.tensor(
                    [sigma_val], device=model.device)

                logits = model.forward_no_diagonal_masking(tokens, sigma)
                probs = model.logits_to_probs(logits)

                target_prob = model.get_target_prob(
                    probs, target_token_id, context_pos)
                if target_prob is None:
                    continue

                probs_at_ctx = probs[0, context_pos]
                log_probs = torch.log(probs_at_ctx + 1e-10)

                entropy = -(probs_at_ctx * log_probs).sum().item()

                sorted_indices = torch.argsort(
                    probs_at_ctx, descending=True)
                rank = (sorted_indices == target_token_id
                        ).nonzero(as_tuple=True)[0].item() + 1

                top_token_id = sorted_indices[0].item()
                top_prob = probs_at_ctx[top_token_id].item()
                top_word = model.tokenizer.decode([top_token_id])

                top2 = torch.topk(probs_at_ctx, k=2).values
                margin = (top2[0] - top2[1]).item()

                top10_mass = torch.topk(
                    probs_at_ctx, k=min(10, len(probs_at_ctx))
                ).values.sum().item()

                results.append({
                    'item': stim['item'],
                    'condition': stim['condition'],
                    'base_condition': stim.get('base_condition', stim['condition']),
                    'sentence': stim['sentence'],
                    'target_word': stim['target_word'],
                    'target_token_id': target_token_id,
                    'target_token_pos': target_pos,
                    'disamb_position': stim['disamb_position'],
                    'ambiguous': stim.get('ambiguous', None),
                    'sigma': sigma_val,
                    'log_sigma': np.log(sigma_val),
                    'target_prob': target_prob,
                    'target_surprisal': compute_surprisal(target_prob),
                    'target_rank': rank,
                    'entropy': entropy,
                    'top_prob': top_prob,
                    'top_word': top_word,
                    'confidence_margin': margin,
                    'top10_mass': top10_mass,
                })

    return pd.DataFrame(results)


def plot_sigma_sweep(df, output_dir, gpt2_df=None):
    """Create visualizations testing two hypotheses about denoising."""
    has_ambig = 'ambiguous' in df.columns and df['ambiguous'].notna().any()
    has_base = 'base_condition' in df.columns
    amb_labels = {0: 'Unambiguous', 1: 'Ambiguous'}
    amb_colors = {0: '#2196F3', 1: '#F44336'}
    cond_colors = {'NPS': '#E91E63', 'NPZ': '#9C27B0', 'MVRR': '#FF9800'}

    # --- Figure 1: Overview (all sentences) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # H1: Entropy vs sigma
    ax = axes[0]
    grouped = df.groupby('sigma')['entropy'].agg(['mean', 'sem'])
    grouped['mean'] = grouped['mean'] * NATS_TO_BITS
    grouped['sem'] = grouped['sem'] * NATS_TO_BITS
    ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['sem'],
                fmt='o-', capsize=3, markersize=4, color='#4CAF50')
    ax.set_xscale('log')
    ax.set_xlabel('Sigma (noise level)', fontsize=12)
    ax.set_ylabel('Distribution Entropy (bits)', fontsize=12)
    ax.set_title('H1: Does denoising reduce uncertainty?', fontsize=13)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)

    # H2: P(target) vs sigma
    ax = axes[1]
    grouped = df.groupby('sigma')['target_prob'].agg(['mean', 'sem'])
    ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['sem'],
                fmt='o-', capsize=3, markersize=4, color='#FF5722')
    ax.set_xscale('log')
    ax.set_xlabel('Sigma (noise level)', fontsize=12)
    ax.set_ylabel('P(target token)', fontsize=12)
    ax.set_title('H2: Does denoising increase P(target)?', fontsize=13)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)

    if gpt2_df is not None and len(gpt2_df) > 0:
        gpt2_entropy = gpt2_df['entropy_bits'].mean()
        axes[0].axhline(gpt2_entropy, color='gray', linestyle=':', linewidth=2, label='GPT-2')
        axes[0].legend()
        gpt2_prob = gpt2_df['target_prob'].mean()
        axes[1].axhline(gpt2_prob, color='gray', linestyle=':', linewidth=2, label='GPT-2')
        axes[1].legend()

    plt.tight_layout()
    path = f'{output_dir}/exp1_sigma_sweep.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved plot: {path}")
    plt.close()

    # --- Figure 2: Ambiguous vs Unambiguous ---
    if has_ambig:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax, metric, ylabel, title in [
            (axes[0], 'entropy', 'Distribution Entropy (bits)',
             'H1: Entropy — Amb vs Unamb'),
            (axes[1], 'target_prob', 'P(target token)',
             'H2: P(target) — Amb vs Unamb'),
        ]:
            for amb_val in sorted(df['ambiguous'].dropna().unique()):
                sub = df[df['ambiguous'] == amb_val]
                grouped = sub.groupby('sigma')[metric].agg(['mean', 'sem'])
                if metric == 'entropy':
                    grouped['mean'] = grouped['mean'] * NATS_TO_BITS
                    grouped['sem'] = grouped['sem'] * NATS_TO_BITS
                label = amb_labels.get(int(amb_val), str(amb_val))
                color = amb_colors.get(int(amb_val))
                ax.plot(grouped.index, grouped['mean'], 'o-',
                        label=label, color=color, markersize=4)
                ax.fill_between(grouped.index,
                                grouped['mean'] - grouped['sem'],
                                grouped['mean'] + grouped['sem'],
                                alpha=0.15, color=color)

            if gpt2_df is not None and len(gpt2_df) > 0:
                gpt2_has_ambig = 'ambiguous' in gpt2_df.columns and gpt2_df['ambiguous'].notna().any()
                if gpt2_has_ambig:
                    for amb_val in sorted(gpt2_df['ambiguous'].dropna().unique()):
                        gpt2_sub = gpt2_df[gpt2_df['ambiguous'] == amb_val]
                        g_label = amb_labels.get(int(amb_val), str(amb_val))
                        g_color = amb_colors.get(int(amb_val))
                        if metric == 'entropy':
                            g_val = gpt2_sub['entropy_bits'].mean()
                        else:
                            g_val = gpt2_sub['target_prob'].mean()
                        ax.axhline(g_val, color=g_color, linestyle=':', linewidth=1.5,
                                   label=f'GPT-2 {g_label}')

            ax.set_xscale('log')
            ax.set_xlabel('Sigma (noise level)', fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(title, fontsize=13)
            ax.invert_xaxis()
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = f'{output_dir}/exp1_ambiguity_effect.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved plot: {path}")
        plt.close()

    # --- Figure 3: By condition ---
    if has_base:
        base_conds = sorted(df['base_condition'].unique())
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax, metric, ylabel, title in [
            (axes[0], 'entropy', 'Distribution Entropy (bits)',
             'H1: Entropy by condition'),
            (axes[1], 'target_prob', 'P(target token)',
             'H2: P(target) by condition'),
        ]:
            for bc in base_conds:
                sub = df[df['base_condition'] == bc]
                grouped = sub.groupby('sigma')[metric].agg(['mean', 'sem'])
                if metric == 'entropy':
                    grouped['mean'] = grouped['mean'] * NATS_TO_BITS
                    grouped['sem'] = grouped['sem'] * NATS_TO_BITS
                color = cond_colors.get(bc)
                ax.plot(grouped.index, grouped['mean'], 'o-',
                        label=bc, color=color, markersize=4)
                ax.fill_between(grouped.index,
                                grouped['mean'] - grouped['sem'],
                                grouped['mean'] + grouped['sem'],
                                alpha=0.15, color=color)

            if gpt2_df is not None and len(gpt2_df) > 0:
                gpt2_has_base = 'base_condition' in gpt2_df.columns
                if gpt2_has_base:
                    for bc in base_conds:
                        gpt2_sub = gpt2_df[gpt2_df['base_condition'] == bc]
                        if len(gpt2_sub) == 0:
                            continue
                        g_color = cond_colors.get(bc)
                        if metric == 'entropy':
                            g_val = gpt2_sub['entropy_bits'].mean()
                        else:
                            g_val = gpt2_sub['target_prob'].mean()
                        ax.axhline(g_val, color=g_color, linestyle=':', linewidth=1.5,
                                   label=f'GPT-2 {bc}')

            ax.set_xscale('log')
            ax.set_xlabel('Sigma (noise level)', fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(title, fontsize=13)
            ax.invert_xaxis()
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = f'{output_dir}/exp1_sigma_by_condition.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved plot: {path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Exp 1: Sigma sweep - how noise level affects target prediction')
    parser.add_argument('--input', type=str,
                        default='SAP_stimuli copy/sap_items_ClassicGP.csv',
                        help='Input CSV with Sentence and disambPosition columns')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--sigmas', type=float, nargs='+', default=None,
                        help='Custom sigma values to sweep')
    parser.add_argument('--gpt2-device', type=str, default='cpu',
                        help='Device for GPT-2 baseline (default: cpu)')
    parser.add_argument('--no-gpt2', action='store_true',
                        help='Skip GPT-2 baseline computation')
    args = parser.parse_args()

    output_dir = create_output_dir(args.output_dir)

    print("=" * 60)
    print(" Experiment 1: Sigma Sweep (Denoising Threshold)")
    print("=" * 60)
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output_dir}/")

    print("\nLoading model...")
    model = SEDDModelWrapper(device=args.device)
    print(f"  Model loaded on {model.device}")

    print("\nLoading stimuli...")
    loader = StimulusLoader(args.input, model.tokenizer)
    stimuli = loader.get_stimuli()
    print(f"  Loaded {len(stimuli)} stimuli from {loader.name}")

    if len(stimuli) == 0:
        print("ERROR: No stimuli loaded. Check CSV format.")
        return 1

    # Show example
    ex = stimuli[0]
    print(f"\n  Example stimulus:")
    print(f"    Sentence: {ex['sentence']}")
    print(f"    Target word: '{ex['target_word']}' at word pos {ex['disamb_position']}")
    print(f"    Token pos: {ex['target_token_pos']}, token ID: {ex['target_token_id']}")

    print("\nRunning sigma sweep...")
    sigmas = args.sigmas if args.sigmas else DEFAULT_SIGMAS
    print(f"  Sweeping {len(sigmas)} sigma values: [{sigmas[0]:.4f} ... {sigmas[-1]:.6f}]")

    df = run_sigma_sweep(model, stimuli, sigmas=sigmas)

    csv_path = f'{output_dir}/exp1_sigma_sweep_{loader.name}.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved results: {csv_path}")
    print(f"  Total rows: {len(df)}")

    # Summary
    print("\n" + "=" * 60)
    print("  Summary by sigma level:")
    print("=" * 60)
    summary = df.groupby('sigma').agg({
        'target_surprisal': ['mean', 'std'],
        'target_rank': 'median',
        'entropy': 'mean',
    }).round(3)
    print(summary.to_string())

    if not args.no_gpt2:
        print("\nLoading GPT-2...")
        gpt2 = GPT2ModelWrapper(device=args.gpt2_device)
        gpt2_df = compute_gpt2_baseline(stimuli, gpt2)
        gpt2_at_target = gpt2_df[gpt2_df['position'] == gpt2_df['target_token_pos'] - 1].copy()
        gpt2_at_target['target_prob'] = 2 ** (-gpt2_at_target['target_surprisal_bits'])
    else:
        gpt2_at_target = None

    print("\nCreating plots...")
    plot_sigma_sweep(df, output_dir, gpt2_df=gpt2_at_target)

    print("\nDone.")
    return 0


if __name__ == '__main__':
    exit(main())
