"""
Experiment 4: Full Position Tracking with Soft Projection

Unlike exp1-3 which evaluate clean tokens at varying sigma, exp4 runs
the actual diffusion process with SOFT left-to-right projection. This
means previous words retain some uncertainty and can keep updating as
later words are denoised — closely modeling human incremental processing
where the role of earlier words is not fully resolved until disambiguating
information arrives.

Tracks two measures across the denoising trajectory for every token
position from 1 to n+3 (n = target position):

(1) Surprisal of the target word (bits)
(2) Entropy of next-token prediction (bits)

Includes GPT-2 baseline for comparison.
Output goes to 'full_tracking_outputs/'.
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from sedd_experiment_utils import (
    SEDDModelWrapper, StimulusLoader, GPT2ModelWrapper,
    compute_surprisal_bits, compute_entropy_bits,
    compute_gpt2_baseline, create_output_dir,
)
from sampling import get_predictor, Denoiser
from model import utils as mutils


# ---------------------------------------------------------------------------
#  Soft projection
# ---------------------------------------------------------------------------
# hard coded function words
FUNCTION_WORDS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'shall',
    'should', 'may', 'might', 'can', 'could', 'must',
    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
    'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'between', 'out', 'off', 'over', 'under', 'again', 'further',
    'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
    'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or',
    'if', 'while', 'although', 'though', 'that', 'which', 'who',
    'whom', 'this', 'these', 'those', 'it', 'its',
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
    'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
    'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'about', 'up',
})


def compute_noise_level(distance, noise_schedule, base_noise):
    """
    Compute the probability of replacing a prior position with noise.

    Args:
        distance: how far before the current processing frontier
                  (1 = immediately before, 2 = two back, ...)
        noise_schedule: scheduling strategy
        base_noise: per-unit-distance noise increment (for 'recency')
                    or peak noise (for legacy schedules)

    Returns:
        Probability of replacing with noise.
        For 'recency': noise INCREASES with distance (farther = noisier),
        matching human recency: recently-read words are clearest.
    """
    if noise_schedule == 'recency':
        # noise = base_noise * d, capped at 0.95
        # e.g. base_noise=0.05 → d=1: 5%, d=2: 10%, d=3: 15%, ...
        return min(0.95, base_noise * distance)
    elif noise_schedule == 'exponential':
        return base_noise * np.exp(-distance / 2)
    elif noise_schedule == 'linear':
        return max(0, base_noise * (1 - distance / 10))
    elif noise_schedule == 'step':
        if distance <= 2:
            return base_noise
        elif distance <= 5:
            return base_noise * 0.3
        else:
            return 0.01
    elif noise_schedule == 'none':
        return 0.0
    else:
        raise ValueError(f"Unknown noise schedule: {noise_schedule}")


def soft_project(x, tokens, target_pos, graph,
                 noise_schedule, base_noise,
                 tokenizer=None, content_bonus=1.0):
    """
    Apply soft projection to all positions before the target.

    Nearby words (small distance) retain more information; farther words
    are noisier — mirroring human recency in working memory.

    When content_bonus < 1.0 and a tokenizer is provided, low-frequency
    content words receive reduced noise (multiplied by content_bonus),
    reflecting the observation that content words leave a stronger trace
    than function words at equivalent distances.
    """
    for pos in range(target_pos):
        distance = target_pos - pos
        noise_level = compute_noise_level(
            distance, noise_schedule, base_noise)

        if tokenizer is not None and content_bonus < 1.0:
            word = tokenizer.decode([tokens[0, pos].item()]).strip().lower()
            if word not in FUNCTION_WORDS:
                noise_level *= content_bonus

        if noise_level > 0 and torch.rand(1).item() < noise_level:
            x[0, pos] = graph.sample_limit(1, 1).to(x.device).squeeze()
        else:
            x[0, pos] = tokens[0, pos]
    return x


# ---------------------------------------------------------------------------
#  Data collection
# ---------------------------------------------------------------------------

def run_full_tracking(model, stimuli, num_steps=256, save_every=4,
                      noise_schedule='recency', base_noise=0.05,
                      content_bonus=1.0):
    """
    Run actual diffusion with soft projection and record target surprisal
    and entropy at every position from 0 to target_pos + 3.

    Unlike exp1-3, the model sees the real noisy state (not clean tokens),
    so predictions reflect the evolving, uncertain representation.
    """
    rows = []

    predictor = get_predictor('analytic')(model.graph, model.noise)
    denoiser = Denoiser(model.graph, model.noise)
    score_fn = mutils.get_score_fn(model.model, train=False, sampling=True)

    eps = 1e-5
    timesteps = torch.linspace(1, eps, num_steps + 1, device=model.device)
    dt = (1 - eps) / num_steps

    tokenizer = model.tokenizer

    for stim in tqdm(stimuli, desc="SEDD soft-projection tracking"):
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

        x = model.graph.sample_limit(1, seq_len).to(model.device)

        with torch.no_grad():
            for i in range(num_steps):
                t = timesteps[i] * torch.ones(
                    1, 1, device=model.device)
                curr_sigma = model.noise(t)[0]

                x = soft_project(x, tokens, target_pos, model.graph,
                                 noise_schedule, base_noise,
                                 tokenizer, content_bonus)

                if i % save_every == 0:
                    logits = model.forward_no_diagonal_masking(
                        x, curr_sigma)
                    probs = model.logits_to_probs(logits)
                    t_val = t[0, 0].item()

                    for p in range(max_pos + 1):
                        target_prob = model.get_target_prob(
                            probs, target_token_id, p)
                        target_surp = (compute_surprisal_bits(target_prob)
                                       if target_prob else None)
                        entropy = compute_entropy_bits(probs[0, p])
                        word = tokenizer.decode(
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

                x = predictor.update_fn(score_fn, x, t, dt)

            # Final denoising step
            x = soft_project(x, tokens, target_pos, model.graph,
                             noise_schedule, base_noise,
                             tokenizer, content_bonus)
            t = timesteps[-1] * torch.ones(
                1, 1, device=model.device)
            x = denoiser.update_fn(score_fn, x, t)

            final_sigma = model.noise(t)[0]
            logits = model.forward_no_diagonal_masking(x, final_sigma)
            probs = model.logits_to_probs(logits)

            for p in range(max_pos + 1):
                target_prob = model.get_target_prob(
                    probs, target_token_id, p)
                target_surp = (compute_surprisal_bits(target_prob)
                               if target_prob else None)
                entropy = compute_entropy_bits(probs[0, p])
                word = tokenizer.decode(
                    [tokens[0, p].item()])

                rows.append({
                    **base_info,
                    'position': p,
                    'word': word,
                    'step': num_steps,
                    'timestep': eps,
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
        description='Exp 4: Full tracking with soft projection + GPT-2')
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
                        help='Number of denoising steps')
    parser.add_argument('--save-every', type=int, default=4,
                        help='Record every N steps')
    parser.add_argument('--noise-schedule', type=str, default='recency',
                        choices=['recency', 'exponential', 'linear', 'step', 'none'],
                        help='Soft projection noise schedule '
                             '(recency = noise increases with distance, '
                             'matching human working-memory decay)')
    parser.add_argument('--base-noise', type=float, default=0.05,
                        help='Per-unit-distance noise rate for recency schedule '
                             '(0.05 → d=1: 5%%, d=2: 10%%, d=3: 15%%, ...)')
    parser.add_argument('--content-bonus', type=float, default=0.5,
                        help='Noise multiplier for content words (< 1.0 means '
                             'content words are retained more than function words; '
                             '1.0 disables the bonus)')
    parser.add_argument('--no-gpt2', action='store_true',
                        help='Skip GPT-2 baseline')
    args = parser.parse_args()

    output_dir = create_output_dir(args.output_dir)

    print("=" * 60)
    print(" Experiment 4: Full Tracking (Soft Projection)")
    print("=" * 60)
    print(f"  Input:          {args.input}")
    print(f"  Steps:          {args.num_steps}, save every {args.save_every}")
    print(f"  Noise schedule: {args.noise_schedule}")
    print(f"  Base noise:     {args.base_noise}")
    print(f"  Content bonus:  {args.content_bonus}")

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

    print("\nRunning SEDD soft-projection tracking...")
    sedd_df = run_full_tracking(
        model, stimuli, args.num_steps, args.save_every,
        noise_schedule=args.noise_schedule,
        base_noise=args.base_noise,
        content_bonus=args.content_bonus,
    )

    tag = f'{args.noise_schedule}_bn{args.base_noise}_cb{args.content_bonus}'
    sedd_path = f'{output_dir}/exp4_full_tracking_{loader.name}_{tag}.csv'
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
