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
    SEDDModelWrapper, StimulusLoader, compute_surprisal, create_output_dir
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


def plot_sigma_sweep(df, output_dir):
    """Create visualizations of sigma sweep results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Target surprisal vs sigma (aggregated across items)
    ax = axes[0, 0]
    grouped = df.groupby('sigma')['target_surprisal'].agg(['mean', 'std'])
    ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                fmt='o-', capsize=3, markersize=4)
    ax.set_xscale('log')
    ax.set_xlabel('Sigma (noise level)', fontsize=12)
    ax.set_ylabel('Target Surprisal (nats)', fontsize=12)
    ax.set_title('How noise affects target prediction', fontsize=13)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)

    # 2. Target rank vs sigma
    ax = axes[0, 1]
    grouped = df.groupby('sigma')['target_rank'].agg(['mean', 'median'])
    ax.plot(grouped.index, grouped['mean'], 'o-', label='Mean rank')
    ax.plot(grouped.index, grouped['median'], 's--', label='Median rank')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Sigma (noise level)', fontsize=12)
    ax.set_ylabel('Target Token Rank', fontsize=12)
    ax.set_title('Rank of target token vs noise', fontsize=13)
    ax.invert_xaxis()
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Entropy vs sigma
    ax = axes[1, 0]
    grouped = df.groupby('sigma')['entropy'].agg(['mean', 'std'])
    ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                fmt='o-', capsize=3, markersize=4, color='green')
    ax.set_xscale('log')
    ax.set_xlabel('Sigma (noise level)', fontsize=12)
    ax.set_ylabel('Distribution Entropy (nats)', fontsize=12)
    ax.set_title('How noise affects prediction uncertainty', fontsize=13)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)

    # 4. Per-item trajectories (subset)
    ax = axes[1, 1]
    unique_items = df['item'].unique()
    sample_items = unique_items[:min(8, len(unique_items))]
    for item in sample_items:
        item_df = df[df['item'] == item].sort_values('sigma', ascending=False)
        cond = item_df['condition'].iloc[0]
        ax.plot(item_df['sigma'], item_df['target_surprisal'],
                'o-', alpha=0.6, markersize=3,
                label=f'Item {item} ({cond})')
    ax.set_xscale('log')
    ax.set_xlabel('Sigma (noise level)', fontsize=12)
    ax.set_ylabel('Target Surprisal (nats)', fontsize=12)
    ax.set_title('Per-item sigma trajectories', fontsize=13)
    ax.invert_xaxis()
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = f'{output_dir}/exp1_sigma_sweep.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Saved plot: {plot_path}")
    plt.close()

    # Paired difference plot: (Ambiguous - Unambiguous) per base condition
    has_ambig = 'ambiguous' in df.columns and df['ambiguous'].notna().any()
    has_base = 'base_condition' in df.columns
    if has_ambig and has_base:
        amb_means = df[df['ambiguous'] == 1].groupby(
            ['base_condition', 'item', 'sigma'])['target_surprisal'].mean()
        unamb_means = df[df['ambiguous'] == 0].groupby(
            ['base_condition', 'item', 'sigma'])['target_surprisal'].mean()
        paired = (amb_means - unamb_means).reset_index(name='diff')
        paired = paired.dropna(subset=['diff'])

        if len(paired) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            colors_bc = {'NPS': '#E91E63', 'NPZ': '#9C27B0', 'MVRR': '#FF9800'}

            # Left: per-condition difference curves
            ax = axes[0]
            for bc in sorted(paired['base_condition'].unique()):
                bc_df = paired[paired['base_condition'] == bc]
                grouped = bc_df.groupby('sigma')['diff'].agg(['mean', 'sem'])
                color = colors_bc.get(bc, None)
                ax.plot(grouped.index, grouped['mean'], 'o-', label=bc,
                        color=color, markersize=5)
                ax.fill_between(grouped.index,
                                grouped['mean'] - grouped['sem'],
                                grouped['mean'] + grouped['sem'],
                                alpha=0.15, color=color)
            ax.axhline(0, color='gray', linestyle='--', linewidth=1)
            ax.set_xscale('log')
            ax.set_xlabel('Sigma (noise level)', fontsize=12)
            ax.set_ylabel('Surprisal difference\n(Ambiguous − Unambiguous)', fontsize=12)
            ax.set_title('Garden path effect by condition', fontsize=13)
            ax.invert_xaxis()
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)

            # Right: overall difference (averaged across conditions)
            ax = axes[1]
            overall = paired.groupby('sigma')['diff'].agg(['mean', 'sem'])
            ax.plot(overall.index, overall['mean'], 'o-', color='black',
                    markersize=5, linewidth=2)
            ax.fill_between(overall.index,
                            overall['mean'] - overall['sem'],
                            overall['mean'] + overall['sem'],
                            alpha=0.2, color='black')
            ax.axhline(0, color='gray', linestyle='--', linewidth=1)
            ax.set_xscale('log')
            ax.set_xlabel('Sigma (noise level)', fontsize=12)
            ax.set_ylabel('Surprisal difference\n(Ambiguous − Unambiguous)', fontsize=12)
            ax.set_title('Overall garden path effect', fontsize=13)
            ax.invert_xaxis()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            cond_path = f'{output_dir}/exp1_sigma_by_condition.png'
            plt.savefig(cond_path, dpi=150, bbox_inches='tight')
            print(f"  Saved plot: {cond_path}")
            plt.close()

    # Key plot: ambiguous vs unambiguous (garden path effect)
    has_ambig = 'ambiguous' in df.columns and df['ambiguous'].notna().any()
    if has_ambig:
        amb_labels = {0: 'Unambiguous', 1: 'Ambiguous'}
        colors = {0: '#2196F3', 1: '#F44336'}

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: overall ambiguous vs unambiguous
        ax = axes[0]
        for amb_val in sorted(df['ambiguous'].dropna().unique()):
            amb_df = df[df['ambiguous'] == amb_val]
            grouped = amb_df.groupby('sigma')['target_surprisal'].agg(
                ['mean', 'sem'])
            ax.plot(grouped.index, grouped['mean'], 'o-',
                    label=amb_labels.get(int(amb_val), str(amb_val)),
                    color=colors.get(int(amb_val), None), markersize=4)
            ax.fill_between(
                grouped.index,
                grouped['mean'] - grouped['sem'],
                grouped['mean'] + grouped['sem'],
                alpha=0.15, color=colors.get(int(amb_val), None))
        ax.set_xscale('log')
        ax.set_xlabel('Sigma (noise level)', fontsize=12)
        ax.set_ylabel('Mean Target Surprisal (nats)', fontsize=12)
        ax.set_title('Garden path effect across noise levels', fontsize=13)
        ax.invert_xaxis()
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Right: ambiguity effect by base condition
        ax = axes[1]
        base_conds = sorted(df['base_condition'].unique()) if 'base_condition' in df.columns else []
        styles = {'NPS': 'o-', 'NPZ': 's--', 'MVRR': '^:'}
        for bc in base_conds:
            for amb_val in sorted(df['ambiguous'].dropna().unique()):
                sub = df[(df['base_condition'] == bc) &
                         (df['ambiguous'] == amb_val)]
                if len(sub) == 0:
                    continue
                grouped = sub.groupby('sigma')['target_surprisal'].mean()
                style = styles.get(bc, 'o-')
                label = f'{bc} {"Amb" if int(amb_val) else "Unamb"}'
                alpha = 1.0 if int(amb_val) else 0.5
                ax.plot(grouped.index, grouped.values, style,
                        label=label, markersize=3, alpha=alpha)
        ax.set_xscale('log')
        ax.set_xlabel('Sigma (noise level)', fontsize=12)
        ax.set_ylabel('Mean Target Surprisal', fontsize=12)
        ax.set_title('By syntactic condition × ambiguity', fontsize=13)
        ax.invert_xaxis()
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        amb_path = f'{output_dir}/exp1_ambiguity_effect.png'
        plt.savefig(amb_path, dpi=150, bbox_inches='tight')
        print(f"  Saved plot: {amb_path}")
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

    print("\nCreating plots...")
    plot_sigma_sweep(df, output_dir)

    print("\nDone.")
    return 0


if __name__ == '__main__':
    exit(main())
