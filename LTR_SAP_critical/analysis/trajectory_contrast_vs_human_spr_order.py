"""Compare integrated trajectory splits (signed) to human SPR hierarchy.

Human prior (strongest SPR effect first among these five):
  MVRR > NPZ > NPS > Agreement (UNAGREE-AGREE) > High attach (AttachHigh-AttachMulti)

Uses Hypothesis C ``hypothesis_c_item_variability_<source>.csv`` ``mean`` column =
per-item mean ``auc_diff(cond_a-cond_b)`` over items (**signed** ∫ over the grid).

**Ranking mode (this run):** sort contrasts by **descending signed mean** (largest
algebraic value first). Same Spearman check as the |mean| version: does the
empirical rank-by-signed-mean order parallel the human strength ladder?

Run:  .venv_analysis/bin/python3 trajectory_contrast_vs_human_spr_order.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy import stats

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))

from analysis_utils import HUMAN_PATTERN_RESULTS_DIR  # noqa: E402


HUMAN_ROWS = [
    ("ClassicGP", "MVRR_AMB-MVRR_UAMB"),
    ("ClassicGP", "NPZ_AMB-NPZ_UAMB"),
    ("ClassicGP", "NPS_AMB-NPS_UAMB"),
    ("Agreement", "UNAGREE-AGREE"),
    ("AttachmentAmbiguity", "AttachHigh-AttachMulti"),
]

LBL = {
    "MVRR_AMB-MVRR_UAMB": "MVRR",
    "NPZ_AMB-NPZ_UAMB": "NPZ ",
    "NPS_AMB-NPS_UAMB": "NPS ",
    "UNAGREE-AGREE": " Agr",
    "AttachHigh-AttachMulti": " HiA",
}


def _main(source: str) -> pd.DataFrame:
    p = HUMAN_PATTERN_RESULTS_DIR / f"hypothesis_c_item_variability_{source}.csv"
    df = pd.read_csv(p)
    out_rows = []
    for var in sorted(df["variable"].unique()):
        vals = []
        ok = True
        for subset, contrast in HUMAN_ROWS:
            m = df[(df["subset"] == subset) & (df["contrast"] == contrast)
                   & (df["variable"] == var)]["mean"].values
            if len(m) != 1 or not pd.notna(float(m[0])):
                ok = False
                break
            vals.append(float(m[0]))
        if not ok:
            continue
        # Rank by descending **signed** ∫ (largest algebraic value = rank 0)
        order_signed = sorted(range(5), key=lambda i: -vals[i])
        empirical_rank_for_contrast = [0] * 5
        for rk, ix in enumerate(order_signed):
            empirical_rank_for_contrast[ix] = rk
        human_prior_rank = list(range(5))
        rho, pr = stats.spearmanr(human_prior_rank, empirical_rank_for_contrast)
        tau, pt = stats.kendalltau(human_prior_rank, empirical_rank_for_contrast)

        order_names = [LBL[HUMAN_ROWS[i][1]] for i in order_signed]
        out_rows.append({
            "source": source,
            "variable": var,
            "spearman_rho_order": float(rho) if rho == rho else float("nan"),
            "spearman_p": float(pr) if pr == pr else float("nan"),
            "kendall_tau": float(tau) if tau == tau else float("nan"),
            "mean_signed_MVRR": vals[0],
            "mean_signed_NPZ": vals[1],
            "mean_signed_NPS": vals[2],
            "mean_signed_Agreement": vals[3],
            "mean_signed_AttachHigh": vals[4],
            "order_signed_large_to_small": ">".join(order_names),
        })
    return pd.DataFrame(out_rows)


if __name__ == "__main__":
    for src in ("bidirectional", "critical_position"):
        t = _main(src)
        print(f"\n### {src} (signed mean ∫)\n")
        if t.empty:
            print("(no rows)")
            continue
        pd.set_option("display.width", 220)
        print(t.to_string(index=False))
        out_csv = HUMAN_PATTERN_RESULTS_DIR / (
            f"trajectory_contrast_signed_vs_human_spr_{src}.csv"
        )
        t.to_csv(out_csv, index=False)
        print(f"\nSaved {out_csv}")
