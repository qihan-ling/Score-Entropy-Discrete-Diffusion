"""Rank Phase 4 metrics by closeness of predicted vs observed paired effects.

Phase 4 writes ``conversion_magnitude_<source>.csv`` (one row per metric ×
human_measure × contrast). The human-pattern report excerpt sorts rows by
``|magnitude_ratio|`` (most extreme scale mismatch), which is **not** the same
as smallest absolute error ``|predicted_mean − observed_mean|``.

This script emits:

  * ``phase4_top3_closest_predicted_SPR_RT_<source>.csv`` — for each
    (subset, contrast), the three model metrics minimizing absolute error vs
    observed SPR RT effect (ms).

  * ``phase4_top3_closest_predicted_all_measures_<source>.csv`` — same but
    grouped by (subset, contrast, human_measure).

  * ``phase4_metric_mean_signed_magnitude_ratio_<source>.csv`` — two columns:
    ``metric`` and ``mean_signed_magnitude_ratio`` (mean of Phase 4
    ``predicted_mean / observed_mean``, signed, over all finite cells).

Reads: ``human_pattern_matching/conversion_magnitude_<source>.csv`` (run
``conversion_magnitude.py`` first).

Run::
  python3 phase4_closest_predicted_ranking.py --both
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import analysis_utils as au


def _mean_signed_magnitude_ratio_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean Phase 4 ``magnitude_ratio`` per metric (finite ratios only)."""
    rows: list[dict] = []
    for metric, g in df.groupby("metric", sort=True):
        r = g["magnitude_ratio"].replace([np.inf, -np.inf], np.nan)
        ok = r.dropna()
        rows.append(
            {
                "metric": metric,
                "mean_signed_magnitude_ratio": float(ok.mean())
                if len(ok)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _run(source: str, out_dir: Path) -> None:
    path = out_dir / f"conversion_magnitude_{source}.csv"
    if not path.exists():
        print(f"Skip {source}: missing {path}")
        return
    df_full = pd.read_csv(path)
    df = df_full[df_full["dropped_low_n"].eq(False)].copy()
    df["abs_err"] = (df["predicted_mean"] - df["observed_mean"]).abs()

    # All human measures
    rows_all: list[dict] = []
    for (subset, contrast, hm), g in df.groupby(
        ["subset", "contrast", "human_measure"], sort=False
    ):
        top = g.sort_values("abs_err", ascending=True).head(3)
        for rank, (_, r) in enumerate(top.iterrows(), 1):
            obs = float(r["observed_mean"])
            rows_all.append(
                {
                    "subset": subset,
                    "contrast": contrast,
                    "human_measure": hm,
                    "rank": rank,
                    "metric": r["metric"],
                    "observed_mean": obs,
                    "predicted_mean": r["predicted_mean"],
                    "abs_err": r["abs_err"],
                    "rel_abs_err": r["abs_err"] / np.abs(obs) if obs != 0 else np.nan,
                    "magnitude_ratio": r["magnitude_ratio"],
                }
            )
    out_all = pd.DataFrame(rows_all)
    p_all = out_dir / f"phase4_top3_closest_predicted_all_measures_{source}.csv"
    au.safe_save_csv(out_all, p_all)
    print(f"Saved {p_all} ({len(out_all)} rows)")

    spr = df[df["human_measure"] == "SPR_RT"].copy()
    rows_spr: list[dict] = []
    for (subset, contrast), g in spr.groupby(["subset", "contrast"], sort=False):
        top = g.sort_values("abs_err", ascending=True).head(3)
        for rank, (_, r) in enumerate(top.iterrows(), 1):
            obs = float(r["observed_mean"])
            rows_spr.append(
                {
                    "subset": subset,
                    "contrast": contrast,
                    "human_measure": "SPR_RT",
                    "rank": rank,
                    "metric": r["metric"],
                    "observed_mean": obs,
                    "predicted_mean": r["predicted_mean"],
                    "abs_err_ms": r["abs_err"],
                    "rel_abs_err": r["abs_err"] / np.abs(obs) if obs != 0 else np.nan,
                    "magnitude_ratio": r["magnitude_ratio"],
                }
            )
    p_spr = out_dir / f"phase4_top3_closest_predicted_SPR_RT_{source}.csv"
    au.safe_save_csv(pd.DataFrame(rows_spr), p_spr)
    print(f"Saved {p_spr} ({len(rows_spr)} rows)")

    ratio_tbl = _mean_signed_magnitude_ratio_table(df_full)
    p_ratio = out_dir / f"phase4_metric_mean_signed_magnitude_ratio_{source}.csv"
    au.safe_save_csv(ratio_tbl, p_ratio)
    print(f"Saved {p_ratio} ({len(ratio_tbl)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    au.add_source_arg(parser)
    parser.add_argument(
        "--both",
        action="store_true",
        help="Run bidirectional and critical_position.",
    )
    parser.add_argument(
        "--out_dir",
        default=str(au.HUMAN_PATTERN_RESULTS_DIR),
        help="Directory containing conversion_magnitude_<source>.csv",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    if args.both:
        for src in ("bidirectional", "critical_position"):
            _run(src, out_dir)
    else:
        _run(args.source, out_dir)


if __name__ == "__main__":
    main()
