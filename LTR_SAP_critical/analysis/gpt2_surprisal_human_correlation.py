"""Correlate GPT-2 pooled paired surprisal Δ vs Phase-1 SPR and ET paired means.

For each of the seven design cells (subset × contrast), GPT-2 takes the per-item
mean surprisal at ``critical + CRITICAL_OFFSET`` (same ROI rule as Phase-1 ET /
``human_patterns``), then pooled cond_a − cond_b (same ``PAIRED_CONTRASTS``).

Human benchmarks come from ``human_patterns.csv``:

- SPR: ``measure == SPR_RT`` (note: RelativeClause SPR is filler-residualized;
  GPT-2 is raw surprisal at critical+2).
- ET measures: ``ffd``, ``gz``, ``gp``, ``tt``, ``regin``, ``regout``.

Correlations: Spearman (and Kendall) across the seven cells (n = 7), signed ×
signed and optionally |GPT-2 × |ET for magnitude alignment.

Outputs under ``human_pattern_matching/``:

- ``gpt2_seven_cells_mean_effect.csv``
- ``gpt2_surprise_vs_human_spr_et.csv``

Run:
  .venv_analysis/bin/python3 gpt2_surprisal_human_correlation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))

import analysis_utils as au

from trajectory_et_rank_correlation import ET_MEASURES  # noqa: E402
from trajectory_et_rank_correlation import SEVEN_CELLS  # noqa: E402
from trajectory_et_rank_correlation import _corr_rows  # noqa: E402
from trajectory_et_rank_correlation import _load_et_matrix  # noqa: E402
from trajectory_et_rank_correlation import _tau_rows  # noqa: E402


def _gpt2_surprise_col(gpt: pd.DataFrame) -> str:
    if "mean_surprisal" in gpt.columns:
        return "mean_surprisal"
    if "surprisal" in gpt.columns:
        return "surprisal"
    raise KeyError(
        "GPT-2 CSV missing surprisal column "
        "(expected mean_surprisal or surprisal). Columns: "
        f"{list(gpt.columns)[:20]}"
    )


def _word_pos_col(gpt: pd.DataFrame) -> str:
    if "word_pos" in gpt.columns:
        return "word_pos"
    if "word_position" in gpt.columns:
        return "word_position"
    raise KeyError("GPT-2 CSV missing word_pos / word_position.")


def gpt2_paired_mean_effect(
    subset: str, cond_a: str, cond_b: str
) -> tuple[float, int]:
    """Pooled mean (cond_a − cond_b) of per-item GPT-2 surprisal at ROI; n items."""
    gpt = au.load_gpt2_surprisals(subset)
    surp = _gpt2_surprise_col(gpt)
    wcol = _word_pos_col(gpt)
    crit_lookup = au.critical_word_position_lookup(subset)
    offset = au.CRITICAL_OFFSET_BY_SUBSET[subset]

    sub = gpt[gpt["condition"].isin([cond_a, cond_b])]
    picked: list[dict] = []
    for (item, cond), grp in sub.groupby(["item", "condition"], dropna=False):
        try:
            ikey = int(item)
        except (TypeError, ValueError):
            continue
        crit = crit_lookup.get((ikey, cond))
        if crit is None:
            continue
        target_pos = int(crit) + int(offset)
        row = grp[grp[wcol] == target_pos]
        if row.empty:
            continue
        picked.append(
            {
                "item": ikey,
                "condition": cond,
                "value": float(row[surp].iloc[0]),
            }
        )

    if not picked:
        return float("nan"), 0

    picked_df = pd.DataFrame(picked)
    diffs = au.paired_differences(picked_df, "value", cond_a, cond_b, ["item"])
    if diffs.size == 0:
        return float("nan"), 0
    return float(np.nanmean(diffs)), int(diffs.size)


def _spr_one_row(
    hp: pd.DataFrame, subset: str, contrast: str
) -> tuple[float, int]:
    sub = hp[
        (hp["subset"] == subset)
        & (hp["contrast"] == contrast)
        & (hp["measure"] == "SPR_RT")
    ]
    if sub.empty:
        raise ValueError(f"Missing SPR_RT {(subset, contrast)}")
    exp_off = 2 if subset == "RelativeClause" else 1
    sub = sub[sub["offset"] == exp_off]
    if len(sub) != 1:
        raise ValueError(f"Ambiguous SPR_RT {(subset, contrast)} rows={len(sub)}")
    n = sub["n_items"].iloc[0]
    try:
        n_i = int(n) if pd.notna(n) else 0
    except (TypeError, ValueError):
        n_i = 0
    return float(sub["mean_effect"].iloc[0]), n_i


def _gpt_seven_cells() -> tuple[pd.DataFrame, pd.Series]:
    rows = []
    vec = []
    for lbl, subset, ctr in SEVEN_CELLS:
        cdict = None
        for c in au.PAIRED_CONTRASTS_BY_SUBSET[subset]:
            if c["name"] == ctr:
                cdict = c
                break
        if cdict is None:
            raise KeyError(f"Contrast {ctr} not in PAIRED_CONTRASTS[{subset}]")
        mu, n = gpt2_paired_mean_effect(subset, cdict["cond_a"], cdict["cond_b"])
        rows.append(
            {
                "cell_key": lbl,
                "subset": subset,
                "contrast": ctr,
                "cond_a": cdict["cond_a"],
                "cond_b": cdict["cond_b"],
                "gpt2_cond_a_minus_b_mean_surprisal": mu,
                "n_items_in_pooled_effect": n,
            }
        )
        vec.append(mu)
    df = pd.DataFrame(rows)
    s = pd.Series(vec, index=[t[0] for t in SEVEN_CELLS], dtype=float)
    return df, s


def main() -> None:
    hcsv = au.HUMAN_PATTERN_RESULTS_DIR / "human_patterns.csv"
    if not hcsv.exists():
        raise FileNotFoundError(f"Missing {hcsv}; run human_patterns.py first.")

    gdf, g_signed = _gpt_seven_cells()
    out_cells = au.HUMAN_PATTERN_RESULTS_DIR / "gpt2_seven_cells_mean_effect.csv"
    gdf.to_csv(out_cells, index=False)
    print(f"Saved {out_cells}")

    hp = pd.read_csv(hcsv)

    spr_vals = []
    spr_n = []
    for lbl, subset, ctr in SEVEN_CELLS:
        v, n = _spr_one_row(hp, subset, ctr)
        spr_vals.append(v)
        spr_n.append(n)
    spr_signed = pd.Series(spr_vals, index=[t[0] for t in SEVEN_CELLS])

    rho_spr, p_spr = _corr_rows(spr_signed, g_signed)
    tau_spr, q_spr = _tau_rows(spr_signed, g_signed)
    rho_m_spr, p_m_spr = _corr_rows(spr_signed.abs(), g_signed.abs())
    rho_pearson_spr, _ = stats.pearsonr(
        spr_signed.values.astype(float), g_signed.values.astype(float)
    )

    et_s, et_a = _load_et_matrix(hcsv)
    lbls = [t[0] for t in SEVEN_CELLS]

    rows_out = [
        {
            "comparison": "SPR_vs_GPT2",
            "human_measure": "SPR_RT",
            "spearman_rho": rho_spr,
            "spearman_p": p_spr,
            "kendall_tau": tau_spr,
            "kendall_p": q_spr,
            "pearson_r": (
                float(rho_pearson_spr) if rho_pearson_spr == rho_pearson_spr else np.nan
            ),
            "spearman_rho_mag": rho_m_spr,
            "spearman_p_mag": p_m_spr,
            "notes": (
                "SPR RC residualized vs filler WP fit; GPT-2 surprisal at raw "
                "critical+2 ROI."
            ),
        }
    ]

    for m in ET_MEASURES:
        e_s = et_s.loc[m].astype(float).reindex(lbls).values
        e_a = et_a.loc[m].astype(float).reindex(lbls).values
        g_np = g_signed.reindex(lbls).astype(float).values

        es = pd.Series(e_s, index=lbls)
        gs = pd.Series(g_np, index=lbls)
        rho_et, p_et = _corr_rows(es, gs)
        tau_et, q_et = _tau_rows(es, gs)
        rho_m, p_m = _corr_rows(pd.Series(e_a), pd.Series(np.abs(g_np)))
        rho_p, _ = stats.pearsonr(es.values.astype(float), gs.values.astype(float))

        rows_out.append(
            {
                "comparison": "ET_vs_GPT2",
                "human_measure": m,
                "spearman_rho": rho_et,
                "spearman_p": p_et,
                "kendall_tau": tau_et,
                "kendall_p": q_et,
                "pearson_r": (
                    float(rho_p) if rho_p == rho_p else np.nan
                ),
                "spearman_rho_mag": rho_m,
                "spearman_p_mag": p_m,
                "notes": "ET pooled means raw at ROI (same Phase-1 rule as GPT-2).",
            }
        )

    out_corr = au.HUMAN_PATTERN_RESULTS_DIR / "gpt2_surprise_vs_human_spr_et.csv"
    pd.DataFrame(rows_out).to_csv(out_corr, index=False)
    print(f"Saved {out_corr}")

    print("\n### GPT-2 seven-cell pooled mean surprisal (cond_a − cond_b):")
    pd.set_option("display.width", 200)
    print(gdf.to_string(index=False))

    print("\n### vs SPR:")
    spr_tbl = pd.DataFrame({"cell": lbls, "SPR_RT": spr_vals, "GPT2": g_signed.values})
    print(spr_tbl.to_string(index=False))
    print(
        f"Spearman ρ = {rho_spr} (p = {p_spr}), Kendall τ = {tau_spr}, "
        f"Pearson r = {rho_pearson_spr:.4f}"
    )

    dm = pd.DataFrame(rows_out)
    dm = dm[dm["human_measure"].isin(["SPR_RT"]) | (dm["comparison"] == "ET_vs_GPT2")]
    print("\n### Spearman ρ (signed ET vs GPT-2):")
    et_only = dm[dm["comparison"] == "ET_vs_GPT2"].sort_values(
        "spearman_rho", ascending=False
    )
    print(
        et_only[["human_measure", "spearman_rho", "spearman_p"]].to_string(index=False)
    )


if __name__ == "__main__":
    main()
