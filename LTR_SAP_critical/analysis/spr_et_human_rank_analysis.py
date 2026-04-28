"""Human benchmark rank alignment across the seven Phase-1 design cells.

**(1)** Compare how well **GPT-2 pooled surprisal Δ** versus **hypothesis‑C
trajectory integrals** (bidirectional vs critical_position, each trajectory
metric) reproduce the **cross-construction SPR ranking** (Spearman ρ over 7
cells). This answers whether GPT-2 “beats” SEDD trajectory sources specifically
for *SPR-derived* construction ordering.

**(2)** Correlate **SPR pooled mean RT effects** versus each **ET measure**
(ffd, gz, …) pooled means at the same ROIs — i.e., do SPR magnitude rankings
across constructions match ET rankings (signed and |effect| variants).

Reads:
  ``human_patterns.csv``, ``gpt2_seven_cells_mean_effect.csv``, and
  ``hypothesis_c_item_variability_{bidirectional,critical_position}.csv``.

Writes:
  ``spr_vs_trajectory_vs_gpt2_rank_alignment.csv`` (Spearman/Kendall *p* + exact
   permutation **p*** for ρ when *n*=7 constructions)
  ``spr_vs_et_measure_rank_alignment.csv`` — Spearman *p* (SciPy approximate)
   and **exact** *n*=7 permutation *p* for signed and |·| SPR–ET rank agreement

Run:
  .venv_analysis/bin/python3 spr_et_human_rank_analysis.py
"""

from __future__ import annotations

import itertools
import math
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
from trajectory_et_rank_correlation import _traj_vectors  # noqa: E402


def _spr_seven_cells(hcsv: pd.DataFrame) -> pd.Series:
    """Signed SPR pooled contrast means (SPR_RT) at canonical offset per subset."""
    cell_keys = [t[0] for t in SEVEN_CELLS]
    vals: list[float] = []
    for _, subset, ctr in SEVEN_CELLS:
        sub = hcsv[
            (hcsv["subset"] == subset)
            & (hcsv["contrast"] == ctr)
            & (hcsv["measure"] == "SPR_RT")
        ].copy()
        exp_off = 2 if subset == "RelativeClause" else 1
        sub = sub[sub["offset"] == exp_off]
        if len(sub) != 1:
            raise ValueError(f"AmbiguousSPR {(subset, ctr)} n={len(sub)}")
        vals.append(float(sub["mean_effect"].iloc[0]))
    return pd.Series(vals, index=cell_keys, dtype=float)


def _gpt2_seven_series() -> tuple[pd.Series, Path]:
    p = au.HUMAN_PATTERN_RESULTS_DIR / "gpt2_seven_cells_mean_effect.csv"
    cell_keys = [t[0] for t in SEVEN_CELLS]
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}; run gpt2_surprisal_human_correlation.py.")
    df = pd.read_csv(p)
    s = pd.Series(
        dict(zip(df["cell_key"], df["gpt2_cond_a_minus_b_mean_surprisal"]))
    ).reindex(cell_keys)
    return s.astype(float), p


def exact_two_sided_spearman_p_permute_x_n7(
    y: np.ndarray, x: np.ndarray
) -> float:
    """Two-sided permutation *p*-value for Spearman ρ with *n*=7 paired rows.

    Null construction: fix construction-ordered ``y`` (human SPR here), take
    the seven observed ``x`` values and assign them uniformly at random across
    constructions (``7!`` shuffles). For each shuffle, Spearman ρ between ``y``
    and permuted ``x``; report the fraction with ``|ρ_perm| ≥ |ρ_obs|``.

    This matches testing "no coupling between construction row and measured
    model score," not a student-t approximation for bivariate Gaussian ranks.
    """
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
        if r_perm != r_perm:
            continue
        if abs(r_perm) >= abs(rho_obs) - 1e-14:
            n_ge += 1

    return float(n_ge) / float(denom)


def main() -> None:
    hp_path = au.HUMAN_PATTERN_RESULTS_DIR / "human_patterns.csv"
    if not hp_path.exists():
        raise FileNotFoundError(hp_path)
    hcsv = pd.read_csv(hp_path)

    spr = _spr_seven_cells(hcsv)
    et_signed, et_abs = _load_et_matrix(hp_path)

    rows_q1 = []
    gpt2_series, _ = _gpt2_seven_series()
    rho_g, pg = _corr_rows(spr.astype(float), gpt2_series.astype(float))
    tau_g, qg = _tau_rows(spr.astype(float), gpt2_series.astype(float))
    pr_g, _ = stats.pearsonr(spr.values, gpt2_series.values)
    px_g = exact_two_sided_spearman_p_permute_x_n7(spr.values, gpt2_series.values)
    rows_q1.append(
        {
            "predictor_family": "GPT2_word_surprisal_roi_pooled_delta",
            "source": "",
            "trajectory_metric": "",
            "spearman_rho_vs_human_SPR": rho_g,
            "spearman_p_scipy": pg,
            "spearman_p_exact_perm7_shuffle_model_across_cells": px_g,
            "kendall_tau_vs_human_SPR": tau_g,
            "kendall_p": qg,
            "pearson_r_vs_human_SPR": float(pr_g),
            "interpretation_notes": (
                "GPT-2 token surprisals (sapbenchmark) at stimulus ROI; "
                "not the SEDD schedule 'surprisal' trajectory column."
            ),
        }
    )

    for source in ("bidirectional", "critical_position"):
        t_s, _ = _traj_vectors(source)
        for var in t_s.index.values:
            traj = t_s.loc[var].astype(float).reindex(spr.index)
            rho_t, pt = _corr_rows(spr, traj)
            tau_t, qt = _tau_rows(spr, traj)
            pr_v, _ = stats.pearsonr(spr.values, traj.values)
            px = exact_two_sided_spearman_p_permute_x_n7(spr.values, traj.values)

            rows_q1.append(
                {
                    "predictor_family": "SEDD_trajectory_signed_integral_hypothesis_C",
                    "source": source,
                    "trajectory_metric": (
                        var  # Collision warning: distinct from GPT2 surprisal
                    ),
                    "spearman_rho_vs_human_SPR": rho_t,
                    "spearman_p_scipy": pt,
                    "spearman_p_exact_perm7_shuffle_model_across_cells": px,
                    "kendall_tau_vs_human_SPR": tau_t,
                    "kendall_p": qt,
                    "pearson_r_vs_human_SPR": float(pr_v),
                    "interpretation_notes": (
                        "Integrated cond_a−cond_b from Hypothesis C; "
                        "`surprisal` here names a trajectory-derived score, "
                        "not GPT-2."
                    ),
                }
            )

    df_q1 = pd.DataFrame(rows_q1).sort_values(
        "spearman_rho_vs_human_SPR",
        ascending=False,
        key=lambda z: np.abs(z.fillna(0)),
    )
    q1_out = au.HUMAN_PATTERN_RESULTS_DIR / "spr_vs_trajectory_vs_gpt2_rank_alignment.csv"
    df_q1.to_csv(q1_out, index=False)

    rows_q2: list[dict] = []
    cell_keys = [t[0] for t in SEVEN_CELLS]

    spr_vec = spr.reindex(cell_keys)
    spr_mag = spr_vec.abs()

    for m in ET_MEASURES:
        e = et_signed.loc[m].reindex(cell_keys).astype(float)
        e_mag = et_abs.loc[m].reindex(cell_keys).astype(float)

        rho_s, p_s = _corr_rows(spr_vec, e)
        tau_s, q_s = _tau_rows(spr_vec, e)
        pr_rs, _ = stats.pearsonr(spr_vec.values, e.values)

        rho_m, p_m = _corr_rows(spr_mag, e_mag)
        tau_m, q_m = _tau_rows(spr_mag, e_mag)
        pr_rm, _ = stats.pearsonr(spr_mag.values, e_mag.values)

        px_s = exact_two_sided_spearman_p_permute_x_n7(spr_vec.values, e.values)
        px_m = exact_two_sided_spearman_p_permute_x_n7(spr_mag.values, e_mag.values)

        rows_q2.append(
            {
                "et_measure": m,
                "spearman_SPR_vs_ET_signed": rho_s,
                "spearman_p_signed_scipy": p_s,
                "spearman_p_signed_exact_perm7_shuffle_ET_across_cells": px_s,
                "kendall_SPR_vs_ET_signed": tau_s,
                "kendall_p_signed": q_s,
                "pearson_SPR_vs_ET_signed": float(pr_rs),
                "spearman_SPR_vs_ET_abs": rho_m,
                "spearman_p_abs_scipy": p_m,
                "spearman_p_abs_exact_perm7_shuffle_ET_across_cells": px_m,
                "kendall_SPR_vs_ET_abs": tau_m,
                "kendall_p_abs": q_m,
                "pearson_SPR_vs_ET_abs": float(pr_rm),
                "notes": (
                    "Same seven cells / offsets as Phase‑1 ET; RC SPR residualized;"
                    " magnitude uses pooled |mean_effect| prior to ρ. Perm7 null: fix"
                    " SPR per construction; shuffle ET pooled means across cells."
                ),
            }
        )

    df_q2 = pd.DataFrame(rows_q2)
    q2_out = au.HUMAN_PATTERN_RESULTS_DIR / "spr_vs_et_measure_rank_alignment.csv"
    df_q2.to_csv(q2_out, index=False)

    # --- Console summary ---
    print("Saved:", q1_out)
    print("Saved:", q2_out)

    print("\n### Q1 — Spearman ρ (human pooled SPR vs model; seven constructions)")
    print(
        "Note: ρ=1 here means SPR and model ranks match cell-by-cell. The "
        "five‑contrast ρ≈0.9 from ``trajectory_contrast_vs_human_spr_order.py`` "
        "compares a **fixed theory ladder** [MVRR,NPZ,NPS,Agr,HiA] to ranks from "
        "five selected trajectory means only (no RC / LowAttach); Agreement and "
        "NPS swap vs that ladder for bidirectional entropy — hence ρ<1."
    )
    top = df_q1.sort_values(
        "spearman_rho_vs_human_SPR",
        ascending=False,
        key=lambda zz: zz.abs(),
    ).head(8)
    print(
        top[
            [
                "predictor_family",
                "source",
                "trajectory_metric",
                "spearman_rho_vs_human_SPR",
                "spearman_p_scipy",
                "spearman_p_exact_perm7_shuffle_model_across_cells",
            ]
        ].to_string(index=False)
    )
    gpt_row = df_q1[df_q1["predictor_family"] == "GPT2_word_surprisal_roi_pooled_delta"]
    bid_ent = df_q1[
        (df_q1["source"] == "bidirectional")
        & (df_q1["trajectory_metric"] == "entropy")
    ]

    print(
        "\nReference (narrow): GPT‑2 pooled surprisal rho(SPR)=",
        f"{gpt_row['spearman_rho_vs_human_SPR'].iloc[0]:.6f}",
        "; bidirectional trajectory entropy rho(SPR)=",
        f"{bid_ent['spearman_rho_vs_human_SPR'].iloc[0]:.6f}",
        sep="",
    )

    print("\n### Q2 — SPR vs ET pooled-effect rank alignment (seven cells)")
    q2_show = df_q2[
        [
            "et_measure",
            "spearman_SPR_vs_ET_signed",
            "spearman_p_signed_scipy",
            "spearman_p_signed_exact_perm7_shuffle_ET_across_cells",
            "spearman_SPR_vs_ET_abs",
            "spearman_p_abs_scipy",
            "spearman_p_abs_exact_perm7_shuffle_ET_across_cells",
        ]
    ]
    print(q2_show.to_string(index=False))
    print("\n(Full table with Kendall/Pearson:", str(q2_out) + ")")


if __name__ == "__main__":
    main()
