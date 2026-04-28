"""Correlate SEDD trajectory integrated Δ vs Phase-1 eye-tracking paired means.

Seven aligned design cells (subset × contrast):

  Agr  Agreement          UNAGREE-AGREE
  MVRR ClassicGP         MVRR_AMB-MVRR_UAMB
  NPZ  ClassicGP         NPZ_AMB-NPZ_UAMB
  NPS  ClassicGP         NPS_AMB-NPS_UAMB
  RC   RelativeClause    RC_Obj-RC_Subj
  HiAt AttachmentAmbiguity AttachHigh-AttachMulti
  LowA AttachmentAmbiguity AttachLow-AttachMulti

ET measures (from ``human_patterns.csv``): ffd, gz, gp, tt, regin, regout.
Offset: 1 for all except RC ET rows (offset 2), matching ``human_patterns_section.md``.

Trajectory: ``hypothesis_c_item_variability_<source>.csv`` ``mean`` column
(signed integrated cond_a−cond_b), same seven keys.

Run:  .venv_analysis/bin/python3 trajectory_et_rank_correlation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy import stats

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))

from analysis_utils import HUMAN_PATTERN_RESULTS_DIR  # noqa: E402

ET_MEASURES = ("ffd", "gz", "gp", "tt", "regin", "regout")

SEVEN_CELLS: list[tuple[str, str, str]] = [
    ("Agr", "Agreement", "UNAGREE-AGREE"),
    ("MVRR", "ClassicGP", "MVRR_AMB-MVRR_UAMB"),
    ("NPZ", "ClassicGP", "NPZ_AMB-NPZ_UAMB"),
    ("NPS", "ClassicGP", "NPS_AMB-NPS_UAMB"),
    ("RC", "RelativeClause", "RC_Obj-RC_Subj"),
    ("HiAt", "AttachmentAmbiguity", "AttachHigh-AttachMulti"),
    ("LowA", "AttachmentAmbiguity", "AttachLow-AttachMulti"),
]


def _et_one_row(
    et: pd.DataFrame, subset: str, contrast: str, measure: str,
) -> float:
    sub = et[(et["subset"] == subset) & (et["contrast"] == contrast)
             & (et["measure"] == measure)]
    if sub.empty:
        raise ValueError(f"Missing ET {(subset, contrast, measure)}")
    if len(sub) > 1:
        exp_off = 2 if subset == "RelativeClause" else 1
        sub = sub[sub["offset"] == exp_off]
    if len(sub) != 1:
        raise ValueError(f"Ambiguous ET {(subset, contrast, measure)} rows={len(sub)}")
    return float(sub["mean_effect"].iloc[0])


def _load_et_matrix(hcsv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (wide_signed, wide_abs). Rows = measures, cols = cell keys."""
    raw = pd.read_csv(hcsv)
    et = raw[raw["measure"].isin(ET_MEASURES)].copy()

    cols = {}
    lbls = [t[0] for t in SEVEN_CELLS]
    for lbl, subset, ctr in SEVEN_CELLS:
        cols[lbl] = []
        for m in ET_MEASURES:
            cols[lbl].append(_et_one_row(et, subset, ctr, m))
    df_s = pd.DataFrame({k: cols[k] for k in lbls}, index=list(ET_MEASURES))
    df_a = df_s.abs()
    return df_s, df_a


def _traj_vectors(source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = HUMAN_PATTERN_RESULTS_DIR / f"hypothesis_c_item_variability_{source}.csv"
    df = pd.read_csv(p)
    vars_ = sorted(df["variable"].unique())
    lbls = [t[0] for t in SEVEN_CELLS]
    signed = {}
    for var in vars_:
        row = []
        for lbl, subset, ctr in SEVEN_CELLS:
            m = df[(df["subset"] == subset) & (df["contrast"] == ctr)
                   & (df["variable"] == var)]["mean"].values
            if len(m) != 1 or not pd.notna(float(m[0])):
                row.append(float("nan"))
            else:
                row.append(float(m[0]))
        signed[var] = row
    s_df = pd.DataFrame(signed, index=lbls).T
    a_df = s_df.abs()
    return s_df, a_df


def _corr_rows(a: pd.Series, b: pd.Series) -> tuple[float, float] | tuple[None, None]:
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce")
    m = aa.notna() & bb.notna()
    if int(m.sum()) < 5:
        return None, None
    rho, p = stats.spearmanr(aa[m].values.astype(float),
                             bb[m].values.astype(float))
    return float(rho), float(p)


def _tau_rows(a: pd.Series, b: pd.Series) -> tuple[float, float] | tuple[None, None]:
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce")
    m = aa.notna() & bb.notna()
    if int(m.sum()) < 5:
        return None, None
    tau, p = stats.kendalltau(aa[m].values.astype(float),
                              bb[m].values.astype(float))
    return float(tau), float(p)


def main() -> None:
    hcsv = HUMAN_PATTERN_RESULTS_DIR / "human_patterns.csv"
    et_s, et_a = _load_et_matrix(hcsv)
    et_s.to_csv(HUMAN_PATTERN_RESULTS_DIR / "et_seven_cells_mean_signed.csv")
    et_a.to_csv(HUMAN_PATTERN_RESULTS_DIR / "et_seven_cells_mean_abs.csv")
    print(f"Saved ET matrices: et_seven_cells_mean_signed/abs.csv ({et_s.shape})")

    for source in ("bidirectional", "critical_position"):
        t_s, t_a = _traj_vectors(source)
        rows_mag = []
        rows_sg = []

        for m in ET_MEASURES:
            e_a = et_a.loc[m].astype(float)
            e_s = et_s.loc[m].astype(float)
            for traj_v in t_s.index.values:
                d_a = t_a.loc[traj_v].astype(float)
                d_s = t_s.loc[traj_v].astype(float)

                rho_m, pm = _corr_rows(e_a, d_a)
                tau_m, qm = _tau_rows(e_a, d_a)

                rho_sg, ps = _corr_rows(e_s, d_s)
                tau_sg, qs = _tau_rows(e_s, d_s)

                rows_mag.append({
                    "et_measure": m,
                    "trajectory_variable": traj_v,
                    "spearman_rho": rho_m,
                    "spearman_p": pm,
                    "kendall_tau": tau_m,
                    "kendall_p": qm,
                })
                rows_sg.append({
                    "et_measure": m,
                    "trajectory_variable": traj_v,
                    "spearman_rho": rho_sg,
                    "spearman_p": ps,
                    "kendall_tau": tau_sg,
                    "kendall_p": qs,
                })

        out_mag = HUMAN_PATTERN_RESULTS_DIR / (
            f"trajectory_et_correlation_mag_{source}.csv"
        )
        out_sg = HUMAN_PATTERN_RESULTS_DIR / (
            f"trajectory_et_correlation_signed_{source}.csv"
        )
        pd.DataFrame(rows_mag).sort_values(
            ["et_measure", "spearman_p"], na_position="last"
        ).to_csv(out_mag, index=False)
        pd.DataFrame(rows_sg).sort_values(
            ["et_measure", "spearman_p"], na_position="last"
        ).to_csv(out_sg, index=False)

        # Top matches |rho| magnitude case
        rdf = pd.DataFrame(rows_mag).dropna(subset=["spearman_rho"])
        rdf["absrho"] = rdf["spearman_rho"].abs()
        top = rdf.sort_values("absrho", ascending=False).head(12)
        print(f"\n### {source}")
        print("|ET| vs |SEDD traj| magnitude Spearman — top |rho|:")
        print(top[["et_measure", "trajectory_variable", "spearman_rho",
                   "spearman_p"]].to_string(index=False))
        print("\nSaved:", out_mag)
        print("Saved:", out_sg)


if __name__ == "__main__":
    main()
