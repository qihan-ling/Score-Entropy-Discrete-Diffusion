"""Phase 4 Cross-Subset Fit Exploration.

Tests whether RC/AttachLow's smaller Phase 4 underprediction gap reflects
shared filler-like item structure (world 1) or a subset-specific metric-RT
relationship (world 2), using a participant-level 80/20 train/test split with
a uniform cross-item OLS for both filler and experimental subsets.

All OLS is ``mean_RT_item ~ alpha + beta * metric_item`` (single intercept,
cross-item rows).  Identification is from cross-item variation — the same
estimand for both training sources, making the comparison clean.

Components
----------
1. Item consistency table — cv, frac_dominant_sign, spearman_rho_spr per
   (subset, contrast, metric).
2. Filler vs experimental-subset metric distribution — KS, MW per
   (subset, metric).
3. Participant 80/20 cross-fit — k=10 repeated splits comparing
   magnitude_ratio_filler vs magnitude_ratio_subset on the same held-out 20 %.
4. World-verdict summary — ratio_improvement and evidence flags.

Metric types
------------
  --metric-type commitment   (default, current run)
  --metric-type trajectory   (future run)

Run::
  python3 phase4_cross_subset_fit.py --both --metric-type commitment
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_THIS = os.path.dirname(os.path.abspath(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)

import analysis_utils as au

load_spr_data = au.load_spr_data

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COMMITMENT_METRICS: tuple[str, ...] = (
    "steps_taken",
    "final_surprisal",
    "final_entropy",
    "final_sampler_entropy",
    "final_sampler_p_target",
    "cumulative_kl",
    "sampler_cumulative_kl",
)

TRAJECTORY_METRICS: tuple[str, ...] = (
    "ent_auc",
    "ent_tail_mean",
    "ent_end",
    "argmax_flips",
    "argmax_settle_step",
    "total_kl",
    "max_sampler_kl_value",
    "sem_lock_step",
    "syn_rank_end",
    "final_syntactic_rank",
    "final_sampler_p_target",
    "sampler_p_target_end",
)

HUMAN_MEASURES = ("SPR_RT",) + tuple(au.ET_MEASURES)
K_SPLITS = 10
MIN_ITEMS_OLS = 5
MIN_PARTICIPANTS_SPLIT = 4

# Subset/contrast → SAP canonical ROI column name in raw SPR
_SPR_ROI_OFFSET = {s: au.CRITICAL_OFFSET_BY_SUBSET[s] for s in au.EXPERIMENTAL_SUBSETS}


# ---------------------------------------------------------------------------
# Helpers: data loading
# ---------------------------------------------------------------------------

def _load_raw_spr(subset: str) -> pd.DataFrame:
    """Raw participant-level SPR rows for the given subset."""
    df = load_spr_data(subset)
    if "MD5" in df.columns and "participant" not in df.columns:
        df["participant"] = df["MD5"]
    elif "participant" not in df.columns:
        raise KeyError(f"No MD5/participant column in SPR data for {subset}")
    return df


def _load_raw_et_long() -> pd.DataFrame:
    """Melt all_wide.csv to long: item, subj, cond, measure, word_position, value."""
    wide = au.load_eye_tracking()
    rows = []
    for measure in au.ET_MEASURES:
        cols = [c for c in wide.columns if c.startswith(measure + "R")]
        if not cols:
            continue
        long = wide.melt(
            id_vars=["item", "subj", "cond"],
            value_vars=cols,
            var_name="roi_col",
            value_name="value",
        )
        long["word_position"] = long["roi_col"].str[len(measure) + 1:].astype(int)
        long["measure"] = measure
        rows.append(long[["item", "subj", "cond", "measure", "word_position", "value"]])
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).dropna(subset=["value"])


def _load_raw_et_filler_long() -> pd.DataFrame:
    """Melt filler_wide.csv to long."""
    wide = au.load_filler_eye_tracking()
    rows = []
    for measure in au.ET_MEASURES:
        cols = [c for c in wide.columns if c.startswith(measure + "R")]
        if not cols:
            continue
        long = wide.melt(
            id_vars=["item", "subj", "cond"] if "cond" in wide.columns else ["item", "subj"],
            value_vars=cols,
            var_name="roi_col",
            value_name="value",
        )
        long["word_position"] = long["roi_col"].str[len(measure) + 1:].astype(int)
        long["measure"] = measure
        id_cols = ["item", "subj", "measure", "word_position", "value"]
        if "cond" in long.columns:
            id_cols.insert(2, "cond")
        rows.append(long[id_cols])
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).dropna(subset=["value"])


# ---------------------------------------------------------------------------
# Helpers: cross-item OLS
# ---------------------------------------------------------------------------

def _cross_item_ols(
    item_metric: np.ndarray,
    item_rt: np.ndarray,
) -> tuple[float, float] | None:
    """Simple OLS: mean_RT ~ alpha + beta * metric_item across items.

    Returns (alpha, beta) or None if insufficient data.
    """
    mask = np.isfinite(item_metric) & np.isfinite(item_rt)
    if mask.sum() < MIN_ITEMS_OLS:
        return None
    x, y = item_metric[mask], item_rt[mask]
    if np.unique(x).size < 2:
        return None
    slope, intercept, *_ = stats.linregress(x, y)
    return float(intercept), float(slope)


def _predict_contrast_effect(
    alpha: float,
    beta: float,
    metric_a: np.ndarray,
    metric_b: np.ndarray,
    items_a: np.ndarray,
    items_b: np.ndarray,
) -> float | None:
    """Predicted paired effect: mean(predicted_a - predicted_b) over shared items."""
    pred_a = pd.Series(alpha + beta * metric_a, index=items_a)
    pred_b = pd.Series(alpha + beta * metric_b, index=items_b)
    common = pred_a.index.intersection(pred_b.index)
    if len(common) < MIN_ITEMS_OLS:
        return None
    return float((pred_a[common] - pred_b[common]).mean())


# ---------------------------------------------------------------------------
# Helpers: participant splits
# ---------------------------------------------------------------------------

def _split_participants(
    participants: np.ndarray, seed: int, frac_train: float = 0.8
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    unique = np.unique(participants)
    rng.shuffle(unique)
    n_train = max(1, int(len(unique) * frac_train))
    return unique[:n_train], unique[n_train:]


def _item_means_from_participants(
    df: pd.DataFrame,
    participant_ids: np.ndarray,
    participant_col: str,
    item_col: str,
    value_col: str,
) -> pd.Series:
    """Per-item mean of value_col for selected participants."""
    sub = df[df[participant_col].isin(participant_ids)]
    return sub.groupby(item_col)[value_col].mean()


# ---------------------------------------------------------------------------
# Filler data preparation
# ---------------------------------------------------------------------------

def _filler_model_metrics(source: str, metrics: tuple[str, ...]) -> pd.DataFrame:
    """Per-filler-item mean of each metric (all word positions averaged).

    Returns DataFrame with item as index and metrics as columns.
    """
    filler = au.load_experiment_filler(source)
    if filler.empty:
        return pd.DataFrame()
    avail = [m for m in metrics if m in filler.columns]
    if not avail:
        return pd.DataFrame()
    return filler.groupby("item")[avail].mean()


def _filler_spr_mean_by_participant() -> pd.DataFrame:
    """Per-(participant, item) mean RT over all word positions for filler SPR.

    Filler items have no critical ROI, so we average across all positions.
    Returns columns: participant, item, RT.
    """
    df = au.load_spr_data("Fillers")
    if "MD5" in df.columns and "participant" not in df.columns:
        df["participant"] = df["MD5"]
    if "participant" not in df.columns:
        return pd.DataFrame()
    return df.groupby(["participant", "item"])["RT"].mean().reset_index()


def _filler_et_item_means(measure: str) -> pd.DataFrame:
    """Per-(subj, item) mean for filler ET at the given measure (all word positions)."""
    long = _load_raw_et_filler_long()
    if long.empty:
        return pd.DataFrame()
    sub = long[long["measure"] == measure]
    if sub.empty:
        return pd.DataFrame()
    return sub.groupby(["subj", "item"])["value"].mean().reset_index()


# ---------------------------------------------------------------------------
# Experimental-subset data preparation
# ---------------------------------------------------------------------------

def _experimental_model_metrics(
    source: str, subset: str, metrics: tuple[str, ...]
) -> pd.DataFrame:
    """Per-(item, condition) model metrics at canonical offset."""
    offset = au.CRITICAL_OFFSET_BY_SUBSET[subset]
    df = au.load_experiment_subset(source, subset)
    if df.empty:
        return pd.DataFrame()
    df = df[df["offset"] == offset].copy()
    avail = [m for m in metrics if m in df.columns]
    if not avail:
        return pd.DataFrame()
    return df[["item", "condition"] + avail].drop_duplicates(
        subset=["item", "condition"]
    )


def _experimental_spr_by_participant(subset: str) -> pd.DataFrame:
    """Per-(participant, item, condition) SPR RT at canonical ROI, filtered to subset conditions."""
    offset = au.CRITICAL_OFFSET_BY_SUBSET[subset]
    raw = _load_raw_spr(subset)
    if "ROI" not in raw.columns:
        warnings.warn(f"No ROI column in SPR for {subset}")
        return pd.DataFrame()
    sub = raw[raw["ROI"] == offset].copy()
    if sub.empty:
        return pd.DataFrame()
    # Map Type → SAP condition labels
    inv_alias = {v: k for k, v in au.SPR_CONDITION_ALIASES.items()}
    if "Type" in sub.columns:
        sub["condition"] = sub["Type"].map(lambda v: inv_alias.get(v, v))
    elif "condition" not in sub.columns:
        warnings.warn(f"No condition/Type column in SPR for {subset}")
        return pd.DataFrame()
    # Keep only conditions relevant to this subset
    valid_conds = set()
    for c in au.PAIRED_CONTRASTS_BY_SUBSET.get(subset, []):
        valid_conds.add(c["cond_a"])
        valid_conds.add(c["cond_b"])
    sub = sub[sub["condition"].isin(valid_conds)]
    return sub[["participant", "item", "condition", "RT"]].copy()


def _experimental_et_by_participant(
    subset: str, measure: str, et_long: pd.DataFrame
) -> pd.DataFrame:
    """Per-(subj, item, cond) ET value at the subset's canonical ROI word position."""
    crit_lookup = au.critical_word_position_lookup(subset)
    offset = au.CRITICAL_OFFSET_BY_SUBSET[subset]
    sub = et_long[et_long["measure"] == measure].copy()
    if sub.empty:
        return pd.DataFrame()
    rows = []
    for (item, cond), grp in sub.groupby(["item", "cond"], dropna=False):
        crit = crit_lookup.get((int(item), str(cond)))
        if crit is None:
            continue
        target_pos = int(crit) + int(offset)
        matched = grp[grp["word_position"] == target_pos]
        if matched.empty:
            continue
        for _, r in matched.iterrows():
            rows.append(
                {"subj": r["subj"], "item": int(item), "condition": str(cond), "value": float(r["value"])}
            )
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Component 1 — Item consistency table
# ---------------------------------------------------------------------------

def compute_item_consistency(
    source: str,
    metrics: tuple[str, ...],
    metric_type: str,
    out_dir: Path,
) -> pd.DataFrame:
    """Build per-(subset, contrast, metric) consistency table.

    Reads pre-computed variability CSVs and per-item SPR correlation CSVs.
    """
    if metric_type == "commitment":
        var_path = au.HUMAN_PATTERN_RESULTS_DIR / f"final_metrics_item_variability_{source}.csv"
        spr_corr_path = au.HUMAN_PATTERN_RESULTS_DIR / f"final_metric_itemwise_spr_correlation_{source}.csv"
    else:
        var_path = au.HUMAN_PATTERN_RESULTS_DIR / f"hypothesis_c_item_variability_{source}.csv"
        spr_corr_path = au.HUMAN_PATTERN_RESULTS_DIR / f"itemwise_spr_auc_diff_correlation.csv"

    if not var_path.exists():
        print(f"  [consistency] Missing {var_path}; skipping.")
        return pd.DataFrame()

    var_df = pd.read_csv(var_path)
    # Normalise column names across the two variability CSVs
    if "variable" in var_df.columns and "metric" not in var_df.columns:
        var_df = var_df.rename(columns={"variable": "metric"})

    spr_df = pd.DataFrame()
    if spr_corr_path.exists():
        spr_df = pd.read_csv(spr_corr_path)
        # trajectory corr csv uses 'variable' instead of 'metric'
        if "variable" in spr_df.columns and "metric" not in spr_df.columns:
            spr_df = spr_df.rename(columns={"variable": "metric"})
        # filter to the source
        if "trajectory_source" in spr_df.columns:
            spr_df = spr_df[spr_df["trajectory_source"] == source]
        elif "source" in spr_df.columns:
            spr_df = spr_df[spr_df["source"] == source]

    rows = []
    for metric in metrics:
        var_sub = var_df[var_df["metric"] == metric]
        if var_sub.empty:
            continue
        for _, vr in var_sub.iterrows():
            rec = {
                "source": source,
                "subset": vr["subset"],
                "contrast": vr["contrast"],
                "metric": metric,
                "n_items": int(vr.get("n", np.nan)),
                "mean_delta": float(vr.get("mean", np.nan)),
                "cv": float(vr.get("cv", np.nan)),
                "frac_dominant_sign": float(vr.get("frac_dominant_sign", np.nan)),
            }
            # Join SPR correlation
            if not spr_df.empty:
                sc = spr_df[
                    (spr_df["subset"] == vr["subset"])
                    & (spr_df["contrast"] == vr["contrast"])
                    & (spr_df["metric"] == metric)
                ]
                if len(sc) == 1:
                    rec["spearman_rho_spr"] = float(sc["spearman_rho"].iloc[0])
                    rec["spearman_p_spr"] = float(sc["spearman_p"].iloc[0])
                else:
                    rec["spearman_rho_spr"] = np.nan
                    rec["spearman_p_spr"] = np.nan
            else:
                rec["spearman_rho_spr"] = np.nan
                rec["spearman_p_spr"] = np.nan
            rows.append(rec)

    out = pd.DataFrame(rows)
    path = out_dir / f"phase4_item_consistency_{metric_type}_{source}.csv"
    au.safe_save_csv(out, path)
    print(f"  [Component 1] Saved {path} ({len(out)} rows)")
    return out


# ---------------------------------------------------------------------------
# Component 2 — Filler vs subset metric distribution
# ---------------------------------------------------------------------------

def compute_filler_vs_subset_dist(
    source: str,
    metrics: tuple[str, ...],
    metric_type: str,
    out_dir: Path,
) -> pd.DataFrame:
    """KS / MW comparison of metric distribution: filler pooled vs each subset."""
    filler_metrics = _filler_model_metrics(source, metrics)
    rows = []
    for subset in au.EXPERIMENTAL_SUBSETS:
        exp_df = _experimental_model_metrics(source, subset, metrics)
        if exp_df.empty:
            continue
        # Collapse conditions: take mean per item across conditions for distribution comparison
        exp_per_item = exp_df.groupby("item")[list(metrics)].mean()
        for metric in metrics:
            if metric not in filler_metrics.columns or metric not in exp_per_item.columns:
                continue
            f_vals = filler_metrics[metric].dropna().values
            s_vals = exp_per_item[metric].dropna().values
            if f_vals.size < 3 or s_vals.size < 3:
                continue
            ks_stat, ks_p = stats.ks_2samp(f_vals, s_vals)
            mw_stat, mw_p = stats.mannwhitneyu(
                f_vals, s_vals, alternative="two-sided"
            )
            f_cv = (
                float(np.std(f_vals, ddof=1) / max(abs(np.mean(f_vals)), 1e-12))
                if f_vals.size > 1
                else np.nan
            )
            s_cv = (
                float(np.std(s_vals, ddof=1) / max(abs(np.mean(s_vals)), 1e-12))
                if s_vals.size > 1
                else np.nan
            )
            rows.append(
                {
                    "source": source,
                    "subset": subset,
                    "metric": metric,
                    "filler_n_items": int(f_vals.size),
                    "subset_n_items": int(s_vals.size),
                    "filler_mean": float(np.mean(f_vals)),
                    "filler_cv": f_cv,
                    "subset_mean": float(np.mean(s_vals)),
                    "subset_cv": s_cv,
                    "ks_stat": float(ks_stat),
                    "ks_p": float(ks_p),
                    "mw_stat": float(mw_stat),
                    "mw_p": float(mw_p),
                }
            )

    out = pd.DataFrame(rows)
    path = out_dir / f"phase4_filler_vs_subset_metric_dist_{metric_type}_{source}.csv"
    au.safe_save_csv(out, path)
    print(f"  [Component 2] Saved {path} ({len(out)} rows)")
    return out


# ---------------------------------------------------------------------------
# Component 3 — Participant 80/20 cross-fit
# ---------------------------------------------------------------------------

def _one_contrast_spr(
    source: str,
    subset: str,
    cond_a: str,
    cond_b: str,
    metrics: tuple[str, ...],
    exp_model: pd.DataFrame,       # (item, condition, *metrics)
    filler_model: pd.DataFrame,    # item-indexed, metric columns
    exp_spr: pd.DataFrame,         # (participant, item, condition, RT)
    filler_spr: pd.DataFrame,      # (participant, item, RT) — all positions averaged
    k: int = K_SPLITS,
) -> list[dict]:
    """Return per-metric rows for one (subset, contrast, SPR_RT) cell."""
    rows = []
    if filler_spr.empty:
        return rows

    for metric in metrics:
        if metric not in exp_model.columns:
            continue
        if filler_model.empty or metric not in filler_model.columns:
            continue

        # Per-item metric values for experimental items (conditions a and b)
        m_a = exp_model[exp_model["condition"] == cond_a].set_index("item")[metric]
        m_b = exp_model[exp_model["condition"] == cond_b].set_index("item")[metric]
        # Filler per-item metric (all positions already averaged)
        m_filler = filler_model[metric].dropna()

        all_exp_participants = exp_spr["participant"].unique()
        all_fill_participants = filler_spr["participant"].unique()

        if (
            len(all_exp_participants) < MIN_PARTICIPANTS_SPLIT
            or len(all_fill_participants) < MIN_PARTICIPANTS_SPLIT
        ):
            continue

        ratio_fill_list: list[float] = []
        ratio_sub_list: list[float] = []

        for seed in range(k):
            # --- Filler 80/20: filler_spr already has per-(participant, item) mean RT ---
            fill_train_ids, _ = _split_participants(all_fill_participants, seed=seed)
            fill_rt = _item_means_from_participants(
                filler_spr, fill_train_ids, "participant", "item", "RT"
            )
            fill_items = fill_rt.index.intersection(m_filler.index)
            if len(fill_items) < MIN_ITEMS_OLS:
                continue
            fit_fill = _cross_item_ols(
                m_filler.loc[fill_items].values,
                fill_rt.loc[fill_items].values,
            )
            if fit_fill is None:
                continue
            a_fill, b_fill = fit_fill

            # --- Experimental 80/20 ---
            exp_train_ids, exp_test_ids = _split_participants(
                all_exp_participants, seed=seed
            )
            if len(exp_test_ids) < 1:
                continue

            # Build per-item mean RT for train participants per condition
            def _mean_rt_by_item(cond: str, ids: np.ndarray) -> pd.Series:
                s = exp_spr[
                    (exp_spr["condition"] == cond) & (exp_spr["participant"].isin(ids))
                ]
                return s.groupby("item")["RT"].mean()

            rt_a_train = _mean_rt_by_item(cond_a, exp_train_ids)
            rt_b_train = _mean_rt_by_item(cond_b, exp_train_ids)
            # Pool conditions for subset OLS (both conditions together)
            rt_all_train = pd.concat([rt_a_train, rt_b_train])
            metric_all = pd.concat([m_a, m_b])
            shared_train = rt_all_train.index.intersection(metric_all.index)
            if len(shared_train) < MIN_ITEMS_OLS:
                continue
            fit_sub = _cross_item_ols(
                metric_all.loc[shared_train].values,
                rt_all_train.loc[shared_train].values,
            )
            if fit_sub is None:
                continue
            a_sub, b_sub = fit_sub

            # --- Test: 20 % experimental participants ---
            rt_a_test = _mean_rt_by_item(cond_a, exp_test_ids)
            rt_b_test = _mean_rt_by_item(cond_b, exp_test_ids)
            common_test = rt_a_test.index.intersection(rt_b_test.index)
            if len(common_test) < MIN_ITEMS_OLS:
                continue
            observed_effect = float(
                (rt_a_test.loc[common_test] - rt_b_test.loc[common_test]).mean()
            )
            if observed_effect == 0:
                continue

            # Apply filler slope to experimental items
            pred_fill = _predict_contrast_effect(
                a_fill, b_fill, m_a.values, m_b.values, m_a.index.values, m_b.index.values
            )
            pred_sub = _predict_contrast_effect(
                a_sub, b_sub, m_a.values, m_b.values, m_a.index.values, m_b.index.values
            )
            if pred_fill is None or pred_sub is None:
                continue

            ratio_fill_list.append(pred_fill / observed_effect)
            ratio_sub_list.append(pred_sub / observed_effect)

        if not ratio_fill_list:
            continue
        rows.append(
            {
                "source": source,
                "subset": subset,
                "contrast": f"{cond_a}-{cond_b}",
                "human_measure": "SPR_RT",
                "metric": metric,
                "magnitude_ratio_filler": float(np.mean(ratio_fill_list)),
                "ratio_filler_std": float(np.std(ratio_fill_list, ddof=1))
                if len(ratio_fill_list) > 1
                else np.nan,
                "magnitude_ratio_subset": float(np.mean(ratio_sub_list)),
                "ratio_subset_std": float(np.std(ratio_sub_list, ddof=1))
                if len(ratio_sub_list) > 1
                else np.nan,
                "n_splits_used": int(len(ratio_fill_list)),
                "n_filler_items": int(len(m_filler.dropna())),
                "n_subset_items": int(len(m_a.dropna()) + len(m_b.dropna())),
            }
        )
    return rows


def _one_contrast_et(
    source: str,
    subset: str,
    cond_a: str,
    cond_b: str,
    measure: str,
    metrics: tuple[str, ...],
    exp_model: pd.DataFrame,
    filler_model: pd.DataFrame,
    exp_et: pd.DataFrame,          # (subj, item, condition, value)
    filler_et_raw: pd.DataFrame,   # (subj, item, value) — all positions mean
    k: int = K_SPLITS,
) -> list[dict]:
    """Per-metric rows for one (subset, contrast, ET-measure) cell."""
    rows = []
    for metric in metrics:
        if metric not in exp_model.columns:
            continue
        if filler_model.empty or metric not in filler_model.columns:
            continue

        m_a = exp_model[exp_model["condition"] == cond_a].set_index("item")[metric]
        m_b = exp_model[exp_model["condition"] == cond_b].set_index("item")[metric]
        m_filler = filler_model[metric].dropna()

        if exp_et.empty or filler_et_raw.empty:
            continue

        all_exp_subjs = exp_et["subj"].unique()
        all_fill_subjs = filler_et_raw["subj"].unique()
        if (
            len(all_exp_subjs) < MIN_PARTICIPANTS_SPLIT
            or len(all_fill_subjs) < MIN_PARTICIPANTS_SPLIT
        ):
            continue

        ratio_fill_list: list[float] = []
        ratio_sub_list: list[float] = []

        for seed in range(k):
            # Filler 80/20
            fill_train_ids, _ = _split_participants(all_fill_subjs, seed=seed)
            fill_et_train = filler_et_raw[filler_et_raw["subj"].isin(fill_train_ids)]
            fill_rt = fill_et_train.groupby("item")["value"].mean()
            fill_items = fill_rt.index.intersection(m_filler.index)
            if len(fill_items) < MIN_ITEMS_OLS:
                continue
            fit_fill = _cross_item_ols(
                m_filler.loc[fill_items].values, fill_rt.loc[fill_items].values
            )
            if fit_fill is None:
                continue
            a_fill, b_fill = fit_fill

            # Experimental 80/20
            exp_train_ids, exp_test_ids = _split_participants(all_exp_subjs, seed=seed)
            if len(exp_test_ids) < 1:
                continue

            def _et_item_means(cond: str, ids: np.ndarray) -> pd.Series:
                s = exp_et[
                    (exp_et["condition"] == cond) & (exp_et["subj"].isin(ids))
                ]
                return s.groupby("item")["value"].mean()

            et_a_train = _et_item_means(cond_a, exp_train_ids)
            et_b_train = _et_item_means(cond_b, exp_train_ids)
            et_all_train = pd.concat([et_a_train, et_b_train])
            metric_all = pd.concat([m_a, m_b])
            shared_train = et_all_train.index.intersection(metric_all.index)
            if len(shared_train) < MIN_ITEMS_OLS:
                continue
            fit_sub = _cross_item_ols(
                metric_all.loc[shared_train].values,
                et_all_train.loc[shared_train].values,
            )
            if fit_sub is None:
                continue
            a_sub, b_sub = fit_sub

            et_a_test = _et_item_means(cond_a, exp_test_ids)
            et_b_test = _et_item_means(cond_b, exp_test_ids)
            common_test = et_a_test.index.intersection(et_b_test.index)
            if len(common_test) < MIN_ITEMS_OLS:
                continue
            observed_effect = float(
                (et_a_test.loc[common_test] - et_b_test.loc[common_test]).mean()
            )
            if observed_effect == 0:
                continue

            pred_fill = _predict_contrast_effect(
                a_fill, b_fill, m_a.values, m_b.values, m_a.index.values, m_b.index.values
            )
            pred_sub = _predict_contrast_effect(
                a_sub, b_sub, m_a.values, m_b.values, m_a.index.values, m_b.index.values
            )
            if pred_fill is None or pred_sub is None:
                continue

            ratio_fill_list.append(pred_fill / observed_effect)
            ratio_sub_list.append(pred_sub / observed_effect)

        if not ratio_fill_list:
            continue
        rows.append(
            {
                "source": source,
                "subset": subset,
                "contrast": f"{cond_a}-{cond_b}",
                "human_measure": measure,
                "metric": metric,
                "magnitude_ratio_filler": float(np.mean(ratio_fill_list)),
                "ratio_filler_std": float(np.std(ratio_fill_list, ddof=1))
                if len(ratio_fill_list) > 1
                else np.nan,
                "magnitude_ratio_subset": float(np.mean(ratio_sub_list)),
                "ratio_subset_std": float(np.std(ratio_sub_list, ddof=1))
                if len(ratio_sub_list) > 1
                else np.nan,
                "n_splits_used": int(len(ratio_fill_list)),
                "n_filler_items": int(len(m_filler.dropna())),
                "n_subset_items": int(len(m_a.dropna()) + len(m_b.dropna())),
            }
        )
    return rows


def compute_cross_fit(
    source: str,
    metrics: tuple[str, ...],
    metric_type: str,
    out_dir: Path,
    human_measures: tuple[str, ...] = HUMAN_MEASURES,
) -> pd.DataFrame:
    """Run k=10 participant 80/20 cross-fit for all subsets × contrasts × metrics."""
    print(f"  [Component 3] Loading data for source={source}...")
    filler_model = _filler_model_metrics(source, metrics)
    filler_spr_raw = _filler_spr_mean_by_participant()

    do_et = any(m != "SPR_RT" for m in human_measures)
    et_long = _load_raw_et_long() if do_et else pd.DataFrame()
    filler_et_cache: dict[str, pd.DataFrame] = {}

    all_rows: list[dict] = []

    for subset in au.EXPERIMENTAL_SUBSETS:
        print(f"    subset={subset}")
        exp_model = _experimental_model_metrics(source, subset, metrics)
        if exp_model.empty:
            print(f"      no model data; skip")
            continue

        for contrast in au.PAIRED_CONTRASTS_BY_SUBSET[subset]:
            cond_a, cond_b = contrast["cond_a"], contrast["cond_b"]
            ctr_name = contrast["name"]
            print(f"      contrast={ctr_name}")

            for hm in human_measures:
                if hm == "SPR_RT":
                    exp_spr = _experimental_spr_by_participant(subset)
                    if exp_spr.empty:
                        continue
                    rows = _one_contrast_spr(
                        source, subset, cond_a, cond_b,
                        metrics, exp_model, filler_model,
                        exp_spr, filler_spr_raw,
                    )
                    all_rows.extend(rows)
                else:
                    if et_long.empty:
                        continue
                    exp_et = _experimental_et_by_participant(subset, hm, et_long)
                    if exp_et.empty:
                        continue
                    if hm not in filler_et_cache:
                        filler_et_cache[hm] = _filler_et_item_means(hm)
                    filler_et = filler_et_cache[hm]
                    if filler_et.empty:
                        continue
                    rows = _one_contrast_et(
                        source, subset, cond_a, cond_b, hm,
                        metrics, exp_model, filler_model,
                        exp_et, filler_et,
                    )
                    all_rows.extend(rows)

    out = pd.DataFrame(all_rows)
    # Fix contrast name to match the canonical convention
    if not out.empty and "contrast" in out.columns:
        ctr_map = {
            f"{c['cond_a']}-{c['cond_b']}": c["name"]
            for s in au.EXPERIMENTAL_SUBSETS
            for c in au.PAIRED_CONTRASTS_BY_SUBSET[s]
        }
        out["contrast"] = out["contrast"].map(lambda x: ctr_map.get(x, x))
    path = out_dir / f"phase4_cross_fit_results_{metric_type}_{source}.csv"
    au.safe_save_csv(out, path)
    print(f"  [Component 3] Saved {path} ({len(out)} rows)")
    return out


# ---------------------------------------------------------------------------
# Component 4 — World verdict summary
# ---------------------------------------------------------------------------

_EASY_SUBSETS = {"RelativeClause", "AttachmentAmbiguity"}
_EASY_CONTRASTS = {"RC_Obj-RC_Subj", "AttachLow-AttachMulti"}
_HARD_SUBSETS = {"ClassicGP"}


def compute_world_verdict(
    cross_fit: pd.DataFrame,
    metric_type: str,
    source: str,
    out_dir: Path,
) -> pd.DataFrame:
    """Derive world 1 / world 2 evidence flags per (subset, metric, human_measure)."""
    if cross_fit.empty:
        return pd.DataFrame()

    # Build ratio_improvement per row
    cf = cross_fit.copy()
    cf["abs_ratio_filler"] = cf["magnitude_ratio_filler"].abs()
    cf["abs_ratio_subset"] = cf["magnitude_ratio_subset"].abs()
    cf["ratio_improvement"] = cf["abs_ratio_subset"] / cf["abs_ratio_filler"].replace(0, np.nan)

    rows = []
    for (metric, hm), grp in cf.groupby(["metric", "human_measure"], sort=False):
        # Separate easy (RC, AttachLow) from hard (ClassicGP, etc.)
        easy = grp[grp["contrast"].isin(_EASY_CONTRASTS)]
        hard = grp[~grp["contrast"].isin(_EASY_CONTRASTS)]

        if easy.empty:
            continue

        avg_ri_easy = float(easy["ratio_improvement"].mean())
        avg_ri_hard = float(hard["ratio_improvement"].mean()) if not hard.empty else np.nan
        avg_fill_easy = float(easy["magnitude_ratio_filler"].abs().mean())
        avg_sub_easy = float(easy["magnitude_ratio_subset"].abs().mean())
        avg_fill_hard = float(hard["magnitude_ratio_filler"].abs().mean()) if not hard.empty else np.nan

        world2 = bool(avg_ri_easy > 2.0 and (np.isnan(avg_ri_hard) or avg_ri_hard < 1.5))
        world1 = bool(0.8 <= avg_ri_easy <= 1.2)
        both_fail = bool(avg_fill_easy < 0.1 and avg_sub_easy < 0.1)

        rows.append(
            {
                "source": source,
                "metric": metric,
                "human_measure": hm,
                "avg_ratio_improvement_easy_contrasts": avg_ri_easy,
                "avg_ratio_improvement_hard_contrasts": avg_ri_hard,
                "avg_magnitude_ratio_filler_easy": avg_fill_easy,
                "avg_magnitude_ratio_subset_easy": avg_sub_easy,
                "avg_magnitude_ratio_filler_hard": avg_fill_hard,
                "world1_evidence": world1,
                "world2_evidence": world2,
                "both_fail": both_fail,
                "n_easy_cells": int(len(easy)),
                "n_hard_cells": int(len(hard)),
            }
        )

    out = pd.DataFrame(rows)
    path = out_dir / f"phase4_world_verdict_{metric_type}_{source}.csv"
    au.safe_save_csv(out, path)
    print(f"  [Component 4] Saved {path} ({len(out)} rows)")
    return out


# ---------------------------------------------------------------------------
# Per-source runner
# ---------------------------------------------------------------------------

def run_source(
    source: str,
    metric_type: str,
    human_measures: tuple[str, ...],
    out_dir: Path,
) -> None:
    metrics = COMMITMENT_METRICS if metric_type == "commitment" else TRAJECTORY_METRICS
    print(f"\n=== source={source}, metric_type={metric_type} ===")
    print(f"  metrics: {metrics}")

    # Component 1
    compute_item_consistency(source, metrics, metric_type, out_dir)

    # Component 2
    compute_filler_vs_subset_dist(source, metrics, metric_type, out_dir)

    # Component 3
    cross_fit = compute_cross_fit(source, metrics, metric_type, out_dir, human_measures)

    # Component 4
    compute_world_verdict(cross_fit, metric_type, source, out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    au.add_source_arg(parser)
    parser.add_argument(
        "--both",
        action="store_true",
        help="Run bidirectional and critical_position.",
    )
    parser.add_argument(
        "--metric-type",
        dest="metric_type",
        choices=("commitment", "trajectory"),
        default="commitment",
        help="Which metric set to use.",
    )
    parser.add_argument(
        "--human-measures",
        dest="human_measures",
        default="SPR_RT",
        help="Comma-separated human measures to run (default: SPR_RT). "
             "Set to 'all' for SPR_RT,ffd,gz,gp,tt,regin,regout.",
    )
    parser.add_argument(
        "--out_dir",
        default=str(au.HUMAN_PATTERN_RESULTS_DIR),
    )
    args = parser.parse_args()

    out_dir = au.ensure_dir(args.out_dir)

    if args.human_measures == "all":
        hm = HUMAN_MEASURES
    else:
        hm = tuple(m.strip() for m in args.human_measures.split(",") if m.strip())

    sources = (
        ("bidirectional", "critical_position") if args.both else (args.source,)
    )
    for src in sources:
        run_source(src, args.metric_type, hm, out_dir)


if __name__ == "__main__":
    main()
