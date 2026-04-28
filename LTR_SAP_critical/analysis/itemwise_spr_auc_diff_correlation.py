"""Item-wise Spearman/Pearson: per-item integrated trajectory Δ vs human SPR Δ.

Uses the same Δ as subset variability Hypothesis C: trapezoidal
``auc_diff(cond_a − cond_b)`` on the interpolated denoising grid (not final
ROI-point metrics).

For each (trajectory ``source``, ``subset``, ``contrast``, ``variable``),
joins per-item ``auc_diff(a-b)`` from ``hypothesis_c_item_auc_diff_<source>.csv``
with per-item pooled SPR ``RT(cond_a) − RT(cond_b)`` at the canonical offset
(:func:`critical offset <analysis_utils.CRITICAL_OFFSET_BY_SUBSET>`).
Relative Clause SPR is **position-residualised** via filler WP fits when the
CSV ``spr_filler_position_fit.csv`` exists (matching Phase 1 SPR).

Reads
------
  ``hypothesis_c_item_auc_diff_{bidirectional,critical_position}.csv``
  (run ``hypothesis_c_minimal_pair_rates.py`` per source).

Writes
------
  ``itemwise_spr_auc_diff_correlation.csv`` — one row per cell × metric;
  BH-FDR over all tests in the pooled table (*q*=0.05).

Run::
  .venv_analysis/bin/python3 itemwise_spr_auc_diff_correlation.py

"""

from __future__ import annotations

import argparse
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
from hypothesis_c_minimal_pair_rates import VARIABLES  # noqa: E402


FDR_Q = 0.05
MIN_ITEMS = 4


def _bh_fdr(pvals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini–Hochberg FDR adjustment; nan p preserved as nan."""
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
    """Per-item ``RT(cond_a) − RT(cond_b)`` (ms) at canonical offset."""
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


def _run_one_source(
    source: str,
    auc_path: Path,
    filler_fit: pd.DataFrame,
) -> pd.DataFrame:
    if not auc_path.exists():
        raise FileNotFoundError(
            f"Missing {auc_path}. Run:\n"
            f"  python hypothesis_c_minimal_pair_rates.py --source {source}"
        )
    auc = pd.read_csv(auc_path)
    need = {"subset", "contrast", "variable", "item", "auc_diff(a-b)"}
    missing = need - set(auc.columns)
    if missing:
        raise KeyError(f"{auc_path} missing columns: {missing}")

    auc["item"] = pd.to_numeric(auc["item"], errors="coerce").astype("Int64")
    auc = auc.dropna(subset=["item", "auc_diff(a-b)"])
    auc["item"] = auc["item"].astype(int)

    rows: list[dict] = []
    for (subset, contrast, var), sub in auc.groupby(
        ["subset", "contrast", "variable"], sort=False,
    ):
        if var not in VARIABLES:
            continue
        rd0 = sub.iloc[0]
        cond_a, cond_b = str(rd0["cond_a"]), str(rd0["cond_b"])

        spr_d = _spr_item_paired_delta_ms(subset, cond_a, cond_b, filler_fit)
        if spr_d.empty:
            rows.append({
                "trajectory_source": source,
                "subset": subset, "contrast": contrast,
                "cond_a": cond_a, "cond_b": cond_b,
                "variable": var, "n_items": 0,
                "spearman_rho": np.nan,
                "spearman_p": np.nan,
                "pearson_r": np.nan, "pearson_p": np.nan,
                "notes": "SPR inner-join yielded no overlapping items.",
            })
            continue

        j = sub.merge(spr_d, on="item", how="inner")
        n = len(j)
        if n < MIN_ITEMS:
            rows.append({
                "trajectory_source": source,
                "subset": subset, "contrast": contrast,
                "cond_a": cond_a, "cond_b": cond_b,
                "variable": var, "n_items": int(n),
                "spearman_rho": np.nan,
                "spearman_p": np.nan,
                "pearson_r": np.nan, "pearson_p": np.nan,
                "notes": f"n<{MIN_ITEMS}",
            })
            continue

        x = j["auc_diff(a-b)"].astype(float).values
        y = j["delta_rt_ms"].astype(float).values
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() < MIN_ITEMS:
            rows.append({
                "trajectory_source": source,
                "subset": subset, "contrast": contrast,
                "cond_a": cond_a, "cond_b": cond_b,
                "variable": var, "n_items": int(finite.sum()),
                "spearman_rho": np.nan,
                "spearman_p": np.nan,
                "pearson_r": np.nan, "pearson_p": np.nan,
                "notes": "finite pairs < MIN_ITEMS",
            })
            continue
        x_, y_ = x[finite], y[finite]
        try:
            rho, pr = stats.spearmanr(x_, y_)
            rp, pp = stats.pearsonr(x_, y_)
        except Exception:
            rho = pr = rp = pp = np.nan
        notes = []
        if subset == "RelativeClause" and (
            filler_fit is not None and not filler_fit.empty
        ):
            notes.append("RC SPR residualized (filler WP fit)")
        else:
            notes.append(
                "RC raw SPR here" if subset == "RelativeClause" else "SPR raw"
            )

        rows.append({
            "trajectory_source": source,
            "subset": subset, "contrast": contrast,
            "cond_a": cond_a, "cond_b": cond_b,
            "variable": var, "n_items": int(finite.sum()),
            "spearman_rho": float(rho) if rho == rho else np.nan,
            "spearman_p": float(pr) if pr == pr else np.nan,
            "pearson_r": float(rp) if rp == rp else np.nan,
            "pearson_p": float(pp) if pp == pp else np.nan,
            "notes": "; ".join(notes),
        })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        default="bidirectional,critical_position",
        help="Comma-separated trajectory roots (hypothesis C `--source`).",
    )
    args = parser.parse_args()
    out_dir = au.ensure_dir(au.HUMAN_PATTERN_RESULTS_DIR)

    filler_fit = _load_filler_fit()

    dfs = []
    for src in [s.strip() for s in args.sources.split(",") if s.strip()]:
        p = out_dir / f"hypothesis_c_item_auc_diff_{src}.csv"
        dfs.append(_run_one_source(src, p, filler_fit))

    df = pd.concat(dfs, ignore_index=True)
    ps = df["spearman_p"].values.astype(float)
    p_fdr, sig = _bh_fdr(ps)
    df["spearman_p_fdr"] = p_fdr
    df["spearman_fdr_sig"] = sig

    out = out_dir / "itemwise_spr_auc_diff_correlation.csv"
    au.safe_save_csv(df, out)
    print(f"Wrote {out} ({len(df)} rows)")

    # Brief preview per source
    for src in sorted(df["trajectory_source"].unique()):
        sub = df[df["trajectory_source"] == src].dropna(
            subset=["spearman_rho"]
        )
        if sub.empty:
            continue
        sub = sub.iloc[sub["spearman_p"].astype(float).argsort()]
        print(f"\n### {src}: lowest spearman_p (preview up to 8)")
        pv = [
            "subset", "contrast", "variable",
            "n_items", "spearman_rho", "spearman_p", "spearman_p_fdr",
        ]
        print(sub[pv].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
