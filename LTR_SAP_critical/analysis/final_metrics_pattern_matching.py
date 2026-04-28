"""Human-pattern analyses using JSON **final commitment** scalars.

For each paired contrast, Δmetric = metric(cond_a) − metric(cond_b) on runs at
canonical ``CRITICAL_OFFSET`` (same stimuli as ∫ Hypothesis C).

Writes under ``analysis/results/human_pattern_matching/`` (suffix ``_<source>.csv``
for ``bidirectional`` / ``critical_position``):

  * ``final_metrics_item_diff_wide_*.csv`` — paired Δ columns ``diff_<metric>``.
  * ``final_metrics_item_variability_*.csv`` — spread of Δ across items (Hypothesis C‑style cells).
  * ``final_metric_seven_cell_mean_signed/abs_*.csv`` — pooled construction means on the seven‑cell lattice.
  * ``final_metric_et_correlation_{signed,mag}_*.csv`` — vs Phase‑1 ET pooled means (same semantics as ``trajectory_et_correlation_*``).
  * ``spr_vs_final_metric_rank_alignment_*.csv`` — human SPR seven‑cells vs pooled final‑metric Δ.
  * ``final_metric_itemwise_spr_correlation_*`` / ``final_metric_itemwise_et_correlation_*``.
  * ``final_metric_vs_human_spr_five_contrast_order_*`` — five‑construction ladder Spearman vs human SPR hierarchy.

Columns named ``trajectory_source`` hold the experimental JSON root label (same convention as Hypothesis C outputs).

Run:
  python3 final_metrics_pattern_matching.py --both

"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_THIS = os.path.dirname(os.path.abspath(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)

import analysis_utils as au

from trajectory_et_rank_correlation import ET_MEASURES  # noqa: E402
from trajectory_et_rank_correlation import SEVEN_CELLS  # noqa: E402
from trajectory_et_rank_correlation import _corr_rows  # noqa: E402
from trajectory_et_rank_correlation import _load_et_matrix  # noqa: E402
from trajectory_et_rank_correlation import _tau_rows  # noqa: E402
from trajectory_contrast_vs_human_spr_order import HUMAN_ROWS  # noqa: E402
from trajectory_contrast_vs_human_spr_order import LBL as LADDER_LBL  # noqa: E402


FINAL_METRICS: tuple[str, ...] = (
    "steps_taken",
    "final_surprisal",
    "final_entropy",
    "final_sampler_entropy",
    "final_sampler_p_target",
    "cumulative_kl",
    "sampler_cumulative_kl",
)

FDR_Q = 0.05
MIN_ITEMS = 4


def _bh_fdr(pvals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out = np.full_like(pvals, np.nan, dtype=float)
    sig = np.zeros_like(pvals, dtype=bool)
    finite = np.isfinite(pvals)
    if not np.any(finite):
        return out, sig
    idx = np.where(finite)[0]
    ps = pvals[idx]
    order = np.argsort(ps)
    m = ps.size
    ranked = ps[order]
    adj = ranked * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    adj_unsorted = np.empty_like(adj)
    adj_unsorted[order] = adj
    out[idx] = adj_unsorted
    sig[idx] = adj_unsorted <= FDR_Q
    return out, sig


def load_canonical_experimental_wide(source: str) -> pd.DataFrame:
    """One row per (subset, condition, item) at canonical offset JSON only."""
    rows: list[pd.DataFrame] = []
    for subset in au.EXPERIMENTAL_SUBSETS:
        sdf = au.load_experiment_subset(source, subset)
        if sdf.empty:
            continue
        want = sdf["subset"].map(
            lambda s: au.CRITICAL_OFFSET_BY_SUBSET.get(s)
        )
        sdf = sdf[sdf["offset"] == want].copy()
        rows.append(sdf)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def paired_final_metric_diffs(df: pd.DataFrame) -> pd.DataFrame:
    """Wide table: one row per (subset, contrast, item) with ``diff_*`` columns."""
    out_rows: list[dict] = []
    for subset in au.EXPERIMENTAL_SUBSETS:
        for contrast in au.PAIRED_CONTRASTS_BY_SUBSET.get(subset, []):
            cond_a, cond_b = contrast["cond_a"], contrast["cond_b"]
            ctr_name = contrast["name"]
            da = df[(df["subset"] == subset) & (df["condition"] == cond_a)][
                ["item"] + list(FINAL_METRICS)
            ].copy()
            db = df[(df["subset"] == subset) & (df["condition"] == cond_b)][
                ["item"] + list(FINAL_METRICS)
            ].copy()
            ren_a = {m: f"{m}_a" for m in FINAL_METRICS}
            ren_b = {m: f"{m}_b" for m in FINAL_METRICS}
            da = da.rename(columns=ren_a)
            db = db.rename(columns=ren_b)
            merged = da.merge(db, on="item", how="inner")
            if merged.empty:
                continue
            for _, row in merged.iterrows():
                rec = {
                    "subset": subset,
                    "contrast": ctr_name,
                    "cond_a": cond_a,
                    "cond_b": cond_b,
                    "item": int(row["item"]),
                }
                for m in FINAL_METRICS:
                    va, vb = row[f"{m}_a"], row[f"{m}_b"]
                    if pd.isna(va) or pd.isna(vb):
                        rec[f"diff_{m}"] = np.nan
                    else:
                        rec[f"diff_{m}"] = float(va) - float(vb)
                out_rows.append(rec)
    return pd.DataFrame(out_rows)


def _spr_one_row(hp: pd.DataFrame, subset: str, contrast: str) -> float:
    sub = hp[
        (hp["subset"] == subset)
        & (hp["contrast"] == contrast)
        & (hp["measure"] == "SPR_RT")
    ]
    exp_off = 2 if subset == "RelativeClause" else 1
    sub = sub[sub["offset"] == exp_off]
    if len(sub) != 1:
        raise ValueError(f"Ambiguous SPR {(subset, contrast)} n={len(sub)}")
    return float(sub["mean_effect"].iloc[0])


def _spr_seven_series(hp: pd.DataFrame) -> pd.Series:
    cell_keys = [t[0] for t in SEVEN_CELLS]
    vals = [_spr_one_row(hp, s, c) for _, s, c in SEVEN_CELLS]
    return pd.Series(vals, index=cell_keys, dtype=float)


def exact_perm7_spearman_p(y: np.ndarray, x: np.ndarray) -> float:
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    if y.size != 7 or x.size != 7:
        return float("nan")
    rho_obs, _ = stats.spearmanr(y, x)
    if rho_obs != rho_obs:
        return float("nan")
    n_ge = 0
    denom = math.factorial(7)
    for perm in itertools.permutations(range(7)):
        xp = x[list(perm)]
        r_perm, _ = stats.spearmanr(y, xp)
        if r_perm == r_perm and abs(r_perm) >= abs(rho_obs) - 1e-14:
            n_ge += 1
    return float(n_ge) / float(denom)


def _load_filler_fit() -> pd.DataFrame:
    p = au.HUMAN_PATTERN_RESULTS_DIR / "spr_filler_position_fit.csv"
    if p.exists():
        return pd.read_csv(p)
    try:
        return au.fit_filler_spr_position_lines(save_to=p)
    except Exception:
        return pd.DataFrame()


def _spr_item_paired_delta_ms(
    subset: str,
    cond_a: str,
    cond_b: str,
    filler_fit: pd.DataFrame,
) -> pd.DataFrame:
    offset = au.CRITICAL_OFFSET_BY_SUBSET[subset]
    use_resid = (
        subset == "RelativeClause"
        and filler_fit is not None
        and not filler_fit.empty
    )
    if use_resid:
        try:
            spr = au.load_spr_rc_residualized(filler_fit)
        except (FileNotFoundError, KeyError):
            spr = au.load_spr_item_means(subset)
    else:
        spr = au.load_spr_item_means(subset)
    spr = spr[spr["offset"] == offset].copy()
    spr["item"] = pd.to_numeric(spr["item"], errors="coerce").astype("Int64")
    spr = spr.dropna(subset=["item"])
    spr["item"] = spr["item"].astype(int)
    a = spr[spr["condition"] == cond_a][["item", "RT"]].rename(
        columns={"RT": "RT_a"}
    )
    b = spr[spr["condition"] == cond_b][["item", "RT"]].rename(
        columns={"RT": "RT_b"}
    )
    m = a.merge(b, on="item", how="inner")
    if m.empty:
        return m
    m["delta_rt_ms"] = m["RT_a"].astype(float) - m["RT_b"].astype(float)
    return m[["item", "delta_rt_ms"]]


def _et_item_paired_delta(
    subset: str,
    cond_a: str,
    cond_b: str,
    measure: str,
    et_long: pd.DataFrame,
) -> pd.DataFrame:
    crit_lookup = au.critical_word_position_lookup(subset)
    offset = au.CRITICAL_OFFSET_BY_SUBSET[subset]
    sub = et_long[
        (et_long["measure"] == measure)
        & (et_long["condition"].isin([cond_a, cond_b]))
    ]
    picked: list[dict] = []
    for (item, cond), grp in sub.groupby(["item", "condition"], dropna=False):
        crit = crit_lookup.get((int(item), cond))
        if crit is None:
            continue
        target_pos = int(crit) + int(offset)
        row = grp[grp["word_position"] == target_pos]
        if row.empty:
            continue
        picked.append(
            {
                "item": int(item),
                "condition": cond,
                "value": float(row["value"].iloc[0]),
            }
        )
    if not picked:
        return pd.DataFrame()
    pdf = pd.DataFrame(picked)
    a = pdf[pdf["condition"] == cond_a][["item", "value"]].rename(
        columns={"value": "v_a"}
    )
    b = pdf[pdf["condition"] == cond_b][["item", "value"]].rename(
        columns={"value": "v_b"}
    )
    m = a.merge(b, on="item", how="inner")
    if m.empty:
        return m
    m["delta_et"] = m["v_a"].astype(float) - m["v_b"].astype(float)
    return m[["item", "delta_et"]]


def _seven_cell_means(diff_df: pd.DataFrame) -> pd.DataFrame:
    """Rows = FINAL_METRICS, cols = seven cell short labels."""
    lbls = [t[0] for t in SEVEN_CELLS]
    mat: dict[str, list[float]] = {m: [] for m in FINAL_METRICS}
    for lbl, subset, ctr in SEVEN_CELLS:
        sub = diff_df[
            (diff_df["subset"] == subset) & (diff_df["contrast"] == ctr)
        ]
        for m in FINAL_METRICS:
            col = f"diff_{m}"
            v = sub[col].dropna().values
            mat[m].append(float(np.nanmean(v)) if v.size else float("nan"))
    return pd.DataFrame(mat, index=lbls).T


def _item_variability_table(diff_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (subset, contrast), sub in diff_df.groupby(
        ["subset", "contrast"], sort=False
    ):
        for m in FINAL_METRICS:
            col = f"diff_{m}"
            vals = sub[col].dropna().values.astype(float)
            if vals.size == 0:
                continue
            summ = au.variability_summary(vals)
            mean_v = summ.get("mean", np.nan)
            if not np.isnan(mean_v) and abs(mean_v) > 1e-12:
                sign = np.sign(mean_v)
                frac_dom = float(np.mean(np.sign(vals) == sign))
            else:
                frac_dom = np.nan
            rows.append(
                {
                    "subset": subset,
                    "contrast": contrast,
                    "metric": m,
                    "frac_dominant_sign": frac_dom,
                    "cv_gt_1": bool(
                        (not np.isnan(summ.get("cv", np.nan)))
                        and summ["cv"] > 1.0
                    ),
                    **summ,
                }
            )
    return pd.DataFrame(rows)


def _et_correlation_tables(
    fm_signed: pd.DataFrame,
    et_s: pd.DataFrame,
    et_a: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lbls = [t[0] for t in SEVEN_CELLS]
    rows_mag: list[dict] = []
    rows_sg: list[dict] = []
    for m_et in ET_MEASURES:
        e_a = et_a.loc[m_et].reindex(lbls).astype(float)
        e_s = et_s.loc[m_et].reindex(lbls).astype(float)
        for m_fm in FINAL_METRICS:
            d_s = fm_signed.loc[m_fm].reindex(lbls).astype(float)
            d_a = d_s.abs()
            rho_m, pm = _corr_rows(e_a, d_a)
            tau_m, qm = _tau_rows(e_a, d_a)
            rho_sg, ps = _corr_rows(e_s, d_s)
            tau_sg, qs = _tau_rows(e_s, d_s)
            rows_mag.append(
                {
                    "et_measure": m_et,
                    "final_metric": m_fm,
                    "spearman_rho": rho_m,
                    "spearman_p": pm,
                    "kendall_tau": tau_m,
                    "kendall_p": qm,
                }
            )
            rows_sg.append(
                {
                    "et_measure": m_et,
                    "final_metric": m_fm,
                    "spearman_rho": rho_sg,
                    "spearman_p": ps,
                    "kendall_tau": tau_sg,
                    "kendall_p": qs,
                }
            )
    return pd.DataFrame(rows_mag), pd.DataFrame(rows_sg)


def _spr_vs_human_spr_rows(
    source: str,
    spr_signed: pd.Series,
    fm_signed: pd.DataFrame,
) -> pd.DataFrame:
    """Seven-cell Spearman: pooled mean Δ per final metric vs human SPR Δ."""
    lbls = [t[0] for t in SEVEN_CELLS]
    spr_mag = spr_signed.abs()

    rows_spr: list[dict] = []
    for m_fm in FINAL_METRICS:
        gm = fm_signed.loc[m_fm].reindex(lbls).astype(float)
        rho, p = _corr_rows(spr_signed, gm)
        tau, q = _tau_rows(spr_signed, gm)
        pr, _ = stats.pearsonr(spr_signed.values.astype(float),
                               gm.values.astype(float))
        px = exact_perm7_spearman_p(
            spr_signed.values.astype(float),
            gm.values.astype(float),
        )
        rho_m, p_m = _corr_rows(spr_mag, gm.abs())
        rows_spr.append(
            {
                "predictor_family": "final_commitment_metric",
                "trajectory_source": source,
                "metric": m_fm,
                "spearman_rho_vs_human_SPR": rho,
                "spearman_p_scipy": p,
                "spearman_p_exact_perm7_shuffle_model_across_cells": px,
                "kendall_tau_vs_human_SPR": tau,
                "kendall_p": q,
                "pearson_r_vs_human_SPR": float(pr) if pr == pr else np.nan,
                "spearman_rho_mag_SPR_vs_mag_metric": rho_m,
                "spearman_p_mag": p_m,
            }
        )

    return pd.DataFrame(rows_spr)


def _itemwise_spr_et(
    source: str,
    diff_df: pd.DataFrame,
    filler_fit: pd.DataFrame,
    et_long: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_spr: list[dict] = []
    rows_et: list[dict] = []
    for (subset, contrast), sub_rows in diff_df.groupby(
        ["subset", "contrast"], sort=False
    ):
        subset = str(subset)
        contrast = str(contrast)
        r0 = sub_rows.iloc[0]
        cond_a, cond_b = str(r0["cond_a"]), str(r0["cond_b"])
        sub_rows = diff_df[
            (diff_df["subset"] == subset)
            & (diff_df["contrast"] == contrast)
        ]
        sd = _spr_item_paired_delta_ms(subset, cond_a, cond_b, filler_fit)

        for m_fm in FINAL_METRICS:
            col = f"diff_{m_fm}"
            m_sub = sub_rows[["item", col]].copy()
            m_sub = m_sub.rename(columns={col: "diff_metric"})
            j = m_sub.merge(sd, on="item", how="inner")
            x = j["diff_metric"].astype(float).values
            y = j["delta_rt_ms"].astype(float).values
            fin = np.isfinite(x) & np.isfinite(y)
            if fin.sum() >= MIN_ITEMS:
                rho, pr_sp = stats.spearmanr(x[fin], y[fin])
                rp, pr_pe = stats.pearsonr(x[fin], y[fin])
            else:
                rho = pr_sp = rp = pr_pe = np.nan
            rows_spr.append(
                {
                    "trajectory_source": source,
                    "subset": subset,
                    "contrast": contrast,
                    "metric": m_fm,
                    "n_items": int(fin.sum()),
                    "spearman_rho": float(rho) if rho == rho else np.nan,
                    "spearman_p": float(pr_sp) if pr_sp == pr_sp else np.nan,
                    "pearson_r": float(rp) if rp == rp else np.nan,
                    "pearson_p": float(pr_pe) if pr_pe == pr_pe else np.nan,
                }
            )

        for m_meas in ET_MEASURES:
            ed = _et_item_paired_delta(
                subset, cond_a, cond_b, m_meas, et_long
            )
            for m_fm in FINAL_METRICS:
                col = f"diff_{m_fm}"
                m_sub = sub_rows[["item", col]].rename(
                    columns={col: "diff_metric"}
                )
                j = m_sub.merge(ed, on="item", how="inner")
                x = j["diff_metric"].astype(float).values
                y = j["delta_et"].astype(float).values
                fin = np.isfinite(x) & np.isfinite(y)
                if fin.sum() >= MIN_ITEMS:
                    rho, pr_sp = stats.spearmanr(x[fin], y[fin])
                    rp, pr_pe = stats.pearsonr(x[fin], y[fin])
                else:
                    rho = pr_sp = rp = pr_pe = np.nan
                rows_et.append(
                    {
                        "trajectory_source": source,
                        "subset": subset,
                        "contrast": contrast,
                        "metric": m_fm,
                        "et_measure": m_meas,
                        "n_items": int(fin.sum()),
                        "spearman_rho": float(rho) if rho == rho else np.nan,
                        "spearman_p": float(pr_sp)
                        if pr_sp == pr_sp
                        else np.nan,
                        "pearson_r": float(rp) if rp == rp else np.nan,
                        "pearson_p": float(pr_pe)
                        if pr_pe == pr_pe
                        else np.nan,
                    }
                )

    dfs = pd.DataFrame(rows_spr)
    dfe = pd.DataFrame(rows_et)
    if not dfs.empty:
        ps = dfs["spearman_p"].values.astype(float)
        pfd, sg = _bh_fdr(ps)
        dfs["spearman_p_fdr"] = pfd
        dfs["spearman_fdr_sig"] = sg
    if not dfe.empty:
        ps = dfe["spearman_p"].values.astype(float)
        pfd, sg = _bh_fdr(ps)
        dfe["spearman_p_fdr"] = pfd
        dfe["spearman_fdr_sig"] = sg
    return dfs, dfe


def _five_ladder_signed_vs_spr(diff_df: pd.DataFrame, source: str) -> pd.DataFrame:
    means: dict[tuple[str, str], dict[str, float]] = {}
    for (subset, contrast), sub in diff_df.groupby(
        ["subset", "contrast"], sort=False
    ):
        mu: dict[str, float] = {}
        for m in FINAL_METRICS:
            v = sub[f"diff_{m}"].dropna().values.astype(float)
            mu[m] = float(np.nanmean(v)) if v.size else np.nan
        means[(subset, contrast)] = mu

    rows: list[dict] = []
    for m_fm in FINAL_METRICS:
        vals: list[float] = []
        ok = True
        for subset, contrast in HUMAN_ROWS:
            key = (subset, contrast)
            if key not in means or m_fm not in means[key]:
                ok = False
                break
            vals.append(means[key][m_fm])
        if not ok or len(vals) != 5:
            continue
        order_signed = sorted(range(5), key=lambda i: -vals[i])
        emp_rank = [0] * 5
        for rk, ix in enumerate(order_signed):
            emp_rank[ix] = rk
        human_prior = list(range(5))
        rho, pr = stats.spearmanr(human_prior, emp_rank)
        tau, pt = stats.kendalltau(human_prior, emp_rank)
        order_names = [LADDER_LBL[HUMAN_ROWS[i][1]] for i in order_signed]
        rows.append(
            {
                "source": source,
                "metric": m_fm,
                "spearman_rho_order": float(rho) if rho == rho else np.nan,
                "spearman_p": float(pr) if pr == pr else np.nan,
                "kendall_tau": float(tau) if tau == tau else np.nan,
                "kendall_p": float(pt) if pt == pt else np.nan,
                "mean_signed_MVRR": vals[0],
                "mean_signed_NPZ": vals[1],
                "mean_signed_NPS": vals[2],
                "mean_signed_Agreement": vals[3],
                "mean_signed_AttachHigh": vals[4],
                "order_signed_large_to_small": ">".join(order_names),
            }
        )
    return pd.DataFrame(rows)


def run_source(source: str, out_dir: Path, hp_csv: Path) -> None:
    wide = load_canonical_experimental_wide(source)
    if wide.empty:
        print(f"No canonical JSON rows for source={source}; skipping.")
        return
    wide = wide.drop_duplicates(
        subset=["subset", "condition", "item"], keep="first"
    )

    diff_df = paired_final_metric_diffs(wide)
    if diff_df.empty:
        print(f"No paired differences for source={source}.")
        return

    ou = au.ensure_dir(out_dir)
    long_path = ou / f"final_metrics_item_diff_wide_{source}.csv"
    au.safe_save_csv(diff_df, long_path)
    print(f"Saved {long_path} ({len(diff_df)} rows)")

    iv_path = ou / f"final_metrics_item_variability_{source}.csv"
    iv = _item_variability_table(diff_df)
    au.safe_save_csv(iv, iv_path)
    print(f"Saved {iv_path}")

    fm_s = _seven_cell_means(diff_df)
    fm_s.to_csv(ou / f"final_metric_seven_cell_mean_signed_{source}.csv")
    fm_s.abs().to_csv(
        ou / f"final_metric_seven_cell_mean_abs_{source}.csv"
    )

    et_s, et_a = _load_et_matrix(hp_csv)
    mag_tab, sg_tab = _et_correlation_tables(fm_s, et_s, et_a)
    mag_tab.to_csv(
        ou / f"final_metric_et_correlation_mag_{source}.csv",
        index=False,
    )
    sg_tab.to_csv(
        ou / f"final_metric_et_correlation_signed_{source}.csv",
        index=False,
    )
    print(f"Saved final_metric_et_correlation_(mag|signed)_{source}.csv")

    hp_df = pd.read_csv(hp_csv)
    spr7 = _spr_seven_series(hp_df)

    rows_spr = _spr_vs_human_spr_rows(source, spr7, fm_s)
    spr_path = ou / f"spr_vs_final_metric_rank_alignment_{source}.csv"
    au.safe_save_csv(rows_spr, spr_path)
    print(f"Saved {spr_path}")

    filler = _load_filler_fit()
    et_long = au.load_et_item_means()

    iw_spr, iw_et = _itemwise_spr_et(
        source, diff_df, filler, et_long
    )
    au.safe_save_csv(
        iw_spr,
        ou / f"final_metric_itemwise_spr_correlation_{source}.csv",
    )
    au.safe_save_csv(
        iw_et,
        ou / f"final_metric_itemwise_et_correlation_{source}.csv",
    )
    print(f"Saved itemwise SPR/ET for {source}")

    lad = _five_ladder_signed_vs_spr(diff_df, source)
    au.safe_save_csv(
        lad,
        ou / (
            f"final_metric_vs_human_spr_five_contrast_order_{source}.csv"
        ),
    )
    print(f"Saved five-contrast SPR order ladder for {source}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Final-commitment Δ vs SPR/ET pattern tests (paired snapshot)."
    )
    parser.add_argument(
        "--source",
        choices=("bidirectional", "critical_position"),
        help="Single experiment JSON root label.",
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Run bidirectional and critical_position.",
    )
    args = parser.parse_args()
    if args.both:
        sources = ("bidirectional", "critical_position")
    elif args.source:
        sources = (args.source,)
    else:
        sources = ("bidirectional", "critical_position")

    hp_csv = au.HUMAN_PATTERN_RESULTS_DIR / "human_patterns.csv"
    if not hp_csv.exists():
        print(f"Missing {hp_csv}; human-pattern ET/SPR lookups required.")
        raise SystemExit(1)
    ou = au.HUMAN_PATTERN_RESULTS_DIR
    for src in sources:
        run_source(src, ou, hp_csv)


if __name__ == "__main__":
    main()