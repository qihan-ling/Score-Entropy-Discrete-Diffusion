"""
Experiment 4: Full Position Tracking

Tracks two measures across the full denoising trajectory for every
token position from position 1 to n+3 (where n = target position):

(1) Surprisal of the target word (bits): how much each position's logits
    predict the disambiguating target token.
(2) Entropy of next-token prediction (bits): overall uncertainty of the
    model's prediction at each position.

Both measures are recorded at each denoising step (varying sigma) using
clean tokens (no actual diffusion noise), plus a GPT-2 baseline column.

Output goes to 'full_tracking_outputs/'.
"""

import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from sedd_experiment_utils import (
    SEDDModelWrapper, StimulusLoader, GPT2ModelWrapper,
    compute_surprisal_bits, compute_entropy_bits,
    compute_gpt2_baseline, create_output_dir,
)


def run_full_tracking(model, stimuli, num_steps=256, save_every=4):
    """
    Sweep sigma values and record target surprisal and entropy at every
    position from 0 to target_pos + 3 (capped at sentence length).

    Returns DataFrame with columns:
        ...stimulus info..., position, word, step, timestep,
        target_surprisal_bits, entropy_bits
    """
    rows = []
    eps = 1e-5
    timesteps = torch.linspace(1, eps, num_steps + 1, device=model.device)

    for stim in tqdm(stimuli, desc="SEDD tracking"):
        tokens = model.tokenize(stim['sentence'])
        seq_len = tokens.shape[1]
        target_token_id = stim['target_token_id']
        target_pos = stim['target_token_pos']
        max_pos = min(target_pos + 3, seq_len - 1)

        base_info = {
            'item': stim['item'],
            'condition': stim['condition'],
            'base_condition': stim.get('base_condition', stim['condition']),
            'sentence': stim['sentence'],
            'target_word': stim['target_word'],
            'target_token_pos': target_pos,
            'disamb_position': stim['disamb_position'],
            'ambiguous': stim.get('ambiguous'),
        }

        with torch.no_grad():
            for i in range(0, num_steps + 1, save_every):
                t = timesteps[i] * torch.ones(
                    1, 1, device=model.device)
                sigma = model.noise(t)[0]

                logits = model.forward_no_diagonal_masking(tokens, sigma)
                probs = model.logits_to_probs(logits)
                t_val = t[0, 0].item()

                for p in range(max_pos + 1):
                    target_prob = model.get_target_prob(
                        probs, target_token_id, p)
                    target_surp = (compute_surprisal_bits(target_prob)
                                   if target_prob else None)

                    entropy = compute_entropy_bits(probs[0, p])
                    word = model.tokenizer.decode(
                        [tokens[0, p].item()])

                    rows.append({
                        **base_info,
                        'position': p,
                        'word': word,
                        'step': i,
                        'timestep': t_val,
                        'target_surprisal_bits': target_surp,
                        'entropy_bits': entropy,
                    })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Heatmap builders
# ---------------------------------------------------------------------------

def _build_pivot(df, metric_col, row_col, col_col, col_descending=True):
    """Build a pivot table for heatmap rendering."""
    valid = df.dropna(subset=[metric_col])
    if len(valid) == 0:
        return None
    pivot = valid.pivot_table(
        index=row_col, columns=col_col,
        values=metric_col, aggfunc='mean')
    pivot = pivot.sort_index(axis=0)
    cols = sorted(pivot.columns, reverse=col_descending)
    return pivot[cols]


def _append_gpt2_col(pivot, gpt2_df, metric_col, row_col):
    """Append a GPT-2 column (keyed by row_col) to a pivot table."""
    if gpt2_df is None or len(gpt2_df) == 0:
        return pivot, False
    gpt2_avg = gpt2_df.groupby(row_col)[metric_col].mean()
    gpt2_vec = gpt2_avg.reindex(pivot.index, fill_value=np.nan)
    pivot = pivot.copy()
    pivot['GPT-2'] = gpt2_vec
    return pivot, True


def _render_heatmap(ax, matrix, row_labels, col_labels, vmin, vmax,
                    xlabel, ylabel, title, cbar_label, highlight_gpt2=False):
    """Render a single heatmap on the given axis."""
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd',
                   interpolation='nearest', vmin=vmin, vmax=vmax)

    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)

    n_cols = len(col_labels)
    tick_step = max(1, n_cols // 10)
    x_ticks = list(range(0, n_cols, tick_step))
    if highlight_gpt2 and (n_cols - 1) not in x_ticks:
        x_ticks.append(n_cols - 1)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(col_labels[i]) for i in x_ticks],
                       fontsize=7, rotation=45)

    if highlight_gpt2:
        ax.axvline(n_cols - 1.5, color='white', linewidth=2)

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    plt.colorbar(im, ax=ax, shrink=0.8, label=cbar_label)


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------

def _plot_heatmap_grid(slices, metric_col, row_col, col_col,
                       gpt2_df, output_path,
                       xlabel, ylabel, cbar_label,
                       col_descending=True,
                       ncols=None):
    """
    Generic: build and render a grid of averaged heatmaps for multiple
    data slices, each with an optional GPT-2 column/row.

    slices: list of (label, sub_df)
    """
    n = len(slices)
    if ncols is None:
        ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    pivots = []
    vmax = 0
    for label, sub_df in slices:
        p = _build_pivot(sub_df, metric_col, row_col, col_col, col_descending)
        has_gpt2 = False
        if p is not None and gpt2_df is not None:
            gpt2_sub = gpt2_df
            if 'ambiguous' in sub_df.columns and label != 'All sentences':
                if label == 'Ambiguous':
                    gpt2_sub = gpt2_df[gpt2_df['ambiguous'] == 1]
                elif label == 'Unambiguous':
                    gpt2_sub = gpt2_df[gpt2_df['ambiguous'] == 0]
                elif label in ('NPS', 'NPZ', 'MVRR'):
                    gpt2_sub = gpt2_df[
                        gpt2_df['base_condition'] == label] if 'base_condition' in gpt2_df.columns else gpt2_df[gpt2_df['condition'] == label]
            p, has_gpt2 = _append_gpt2_col(p, gpt2_sub, metric_col, row_col)
        pivots.append((label, p, has_gpt2))
        if p is not None:
            vmax = max(vmax, np.nanmax(p.values))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 5 * nrows),
                             squeeze=False)

    for idx, (label, p, has_gpt2) in enumerate(pivots):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        if p is None:
            ax.set_title(f'{label}\n(no data)')
            ax.axis('off')
            continue

        row_labels = [str(int(x) + 1) if isinstance(x, (int, float, np.integer, np.floating)) else str(x)
                      for x in p.index]
        col_labels = []
        for x in p.columns:
            if isinstance(x, str):
                col_labels.append(x)
            else:
                col_labels.append(f'{x:.2f}')

        _render_heatmap(ax, p.values, row_labels, col_labels,
                        vmin=0, vmax=vmax,
                        xlabel=xlabel, ylabel=ylabel,
                        title=label, cbar_label=cbar_label,
                        highlight_gpt2=has_gpt2)

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_full_tracking(sedd_df, gpt2_df, output_dir):
    """Create all exp4 heatmap figures."""
    if len(sedd_df) == 0:
        print("  No data to plot.")
        return

    has_ambig = ('ambiguous' in sedd_df.columns and
                 sedd_df['ambiguous'].notna().any())
    has_base = 'base_condition' in sedd_df.columns

    # --- (1) Surprisal heatmaps: position (y) × timestep (x) ---
    # Rows = position, Columns = timestep
    surp_slices = [('All sentences', sedd_df)]
    if has_ambig:
        surp_slices.append(('Ambiguous',
                            sedd_df[sedd_df['ambiguous'] == 1]))
        surp_slices.append(('Unambiguous',
                            sedd_df[sedd_df['ambiguous'] == 0]))

    _plot_heatmap_grid(
        surp_slices, 'target_surprisal_bits',
        row_col='position', col_col='timestep',
        gpt2_df=gpt2_df,
        output_path=f'{output_dir}/exp4_surprisal_heatmap.png',
        xlabel='Timestep (noise → clean)',
        ylabel='Token position (1-indexed)',
        cbar_label='Surprisal (bits)',
    )

    if has_base:
        base_conds = sorted(sedd_df['base_condition'].unique())
        cond_slices = [(bc, sedd_df[sedd_df['base_condition'] == bc])
                       for bc in base_conds]
        _plot_heatmap_grid(
            cond_slices, 'target_surprisal_bits',
            row_col='position', col_col='timestep',
            gpt2_df=gpt2_df,
            output_path=f'{output_dir}/exp4_surprisal_by_condition.png',
            xlabel='Timestep (noise → clean)',
            ylabel='Token position (1-indexed)',
            cbar_label='Surprisal (bits)',
        )

    # --- (2) Entropy heatmaps: timestep (y) × position (x) ---
    # Transposed: Rows = timestep, Columns = position
    ent_slices = [('All sentences', sedd_df)]
    if has_ambig:
        ent_slices.append(('Ambiguous',
                           sedd_df[sedd_df['ambiguous'] == 1]))
        ent_slices.append(('Unambiguous',
                           sedd_df[sedd_df['ambiguous'] == 0]))

    _plot_heatmap_grid(
        ent_slices, 'entropy_bits',
        row_col='timestep', col_col='position',
        gpt2_df=gpt2_df,
        output_path=f'{output_dir}/exp4_entropy_heatmap.png',
        xlabel='Token position (1-indexed)',
        ylabel='Timestep (noise → clean)',
        cbar_label='Entropy (bits)',
        col_descending=False,
    )

    if has_base:
        base_conds = sorted(sedd_df['base_condition'].unique())
        cond_slices = [(bc, sedd_df[sedd_df['base_condition'] == bc])
                       for bc in base_conds]
        _plot_heatmap_grid(
            cond_slices, 'entropy_bits',
            row_col='timestep', col_col='position',
            gpt2_df=gpt2_df,
            output_path=f'{output_dir}/exp4_entropy_by_condition.png',
            xlabel='Token position (1-indexed)',
            ylabel='Timestep (noise → clean)',
            cbar_label='Entropy (bits)',
            col_descending=False,
        )


def main():
    parser = argparse.ArgumentParser(
        description='Exp 4: Full position tracking with GPT-2 comparison')
    parser.add_argument('--input', type=str,
                        default='SAP_stimuli copy/sap_items_ClassicGP.csv',
                        help='Input CSV')
    parser.add_argument('--output-dir', type=str,
                        default='full_tracking_outputs')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for SEDD model')
    parser.add_argument('--gpt2-device', type=str, default='cpu',
                        help='Device for GPT-2 model')
    parser.add_argument('--num-steps', type=int, default=256,
                        help='Number of denoising steps to sweep')
    parser.add_argument('--save-every', type=int, default=4,
                        help='Record every N steps')
    parser.add_argument('--no-gpt2', action='store_true',
                        help='Skip GPT-2 baseline')
    args = parser.parse_args()

    output_dir = create_output_dir(args.output_dir)

    print("=" * 60)
    print(" Experiment 4: Full Position Tracking")
    print("=" * 60)
    print(f"  Input: {args.input}")
    print(f"  Steps: {args.num_steps}, save every {args.save_every}")

    print("\nLoading SEDD model...")
    model = SEDDModelWrapper(device=args.device)
    print(f"  SEDD loaded on {model.device}")

    print("\nLoading stimuli...")
    loader = StimulusLoader(args.input, model.tokenizer)
    stimuli = loader.get_stimuli()
    print(f"  Loaded {len(stimuli)} stimuli from {loader.name}")

    if len(stimuli) == 0:
        print("ERROR: No stimuli loaded.")
        return 1

    ex = stimuli[0]
    print(f"\n  Example: {ex['sentence']}")
    print(f"    Target: '{ex['target_word']}' at pos {ex['target_token_pos']}")

    print("\nRunning SEDD full tracking...")
    sedd_df = run_full_tracking(
        model, stimuli, args.num_steps, args.save_every)

    sedd_path = f'{output_dir}/exp4_full_tracking_{loader.name}.csv'
    sedd_df.to_csv(sedd_path, index=False)
    print(f"  Saved: {sedd_path} ({len(sedd_df)} rows)")

    gpt2_df = None
    if not args.no_gpt2:
        print("\nLoading GPT-2...")
        gpt2 = GPT2ModelWrapper(device=args.gpt2_device)
        print(f"  GPT-2 loaded on {gpt2.device}")

        print("Running GPT-2 baseline...")
        gpt2_df = compute_gpt2_baseline(stimuli, gpt2)
        gpt2_path = f'{output_dir}/exp4_gpt2_baseline_{loader.name}.csv'
        gpt2_df.to_csv(gpt2_path, index=False)
        print(f"  Saved: {gpt2_path} ({len(gpt2_df)} rows)")

    print("\nCreating plots...")
    plot_full_tracking(sedd_df, gpt2_df, output_dir)

    print("\nDone.")
    return 0


if __name__ == '__main__':
    exit(main())
