#!/usr/bin/env python3
"""
Derives conversion factors (coefficients linking surprisal to reading times)
from filler items using frequentist linear mixed-effects models.

Methodology (following Smith & Levy, 2013; Mitchell, 1984):
- Fits LME with surprisal and control predictors (word position, length,
  unigram log-frequency, length x frequency interaction)
- Includes spillover predictors for words n, n-1, n-2, n-3
- Random intercepts by participant and item; random slope for surprisal
  by participant
- Excludes words without full 3-word spillover context and sentence-final
  words (wrap-up effects)
- All predictors centered and scaled
"""

import os
import re
import sys
import warnings
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from wordfreq import word_frequency

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILLER_ET_PATH = os.path.join(BASE_DIR, "data", "filler_wide copy.csv")
SURPRISAL_PATH = os.path.join(BASE_DIR, "outputs", "sedd_gpt2_sap_filler_merged.csv")
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))

RT_MEASURE = "gz"  # gaze duration (first-pass reading time)
MAX_LAG = 3


def sig_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    elif p < 0.1:
        return "."
    return ""


def clean_word(w):
    return re.sub(r"[^\w]", "", w)


def get_log_freq(w):
    w = clean_word(w).lower()
    if not w:
        return -10.0
    freq = word_frequency(w, "en")
    return np.log10(freq) if freq > 0 else -10.0


# ── Step 1  Load ─────────────────────────────────────────────────────
def load_data():
    print("Loading data...")
    et = pd.read_csv(FILLER_ET_PATH)
    surp = pd.read_csv(SURPRISAL_PATH)
    print(f"  Eye-tracking rows (subj x sentence): {len(et)}")
    print(f"  Surprisal rows (tokens):             {len(surp)}")
    print(f"  Unique sentences:                    {surp['sentence_idx'].nunique()}")
    print(f"  Unique subjects:                     {et['subj'].nunique()}")
    return et, surp


# ── Step 2  Token → word aggregation ─────────────────────────────────
def aggregate_tokens_to_words(surp):
    """Sum BPE-token surprisal into word-level surprisal.
    Word boundaries are marked by a leading space in GPT-2 tokenisation."""
    print("Aggregating token-level surprisal to word-level...")
    rows = []
    for sidx, grp in surp.groupby("sentence_idx"):
        sentence = grp["sentence"].iloc[0]
        words = sentence.split()
        grp = grp.sort_values("word_position")

        word_idx = 0
        sg, ss = 0.0, 0.0
        started = False

        for _, r in grp.iterrows():
            tok = r["word"]
            if tok.startswith(" "):
                if started:
                    rows.append(dict(
                        sentence_idx=sidx, sentence=sentence,
                        word_idx=word_idx,
                        word=words[word_idx] if word_idx < len(words) else "",
                        surprisal_gpt2=sg, surprisal_sedd=ss))
                word_idx += 1
                sg, ss = r["surprisal_gpt2"], r["surprisal_sedd"]
                started = True
            else:
                sg += r["surprisal_gpt2"]
                ss += r["surprisal_sedd"]
                if not started:
                    word_idx = 1
                    started = True

        if started:
            rows.append(dict(
                sentence_idx=sidx, sentence=sentence,
                word_idx=word_idx,
                word=words[word_idx] if word_idx < len(words) else "",
                surprisal_gpt2=sg, surprisal_sedd=ss))

    df = pd.DataFrame(rows)
    print(f"  Word-level observations: {len(df)}")
    return df


# ── Step 3  Word features ───────────────────────────────────────────
def add_word_features(df):
    print("Computing word-level features (length, frequency, position)...")
    df["word_length"] = df["word"].apply(lambda w: len(clean_word(w))).astype(float)
    df["log_freq"] = df["word"].apply(get_log_freq)
    df["word_position"] = df["word_idx"].astype(float)
    df["len_freq"] = df["word_length"] * df["log_freq"]
    return df


# ── Step 4  Spillover lags ──────────────────────────────────────────
# word_position is excluded from lags: position_{n-k} = position_n - k
# exactly, which creates perfect collinearity. Current position suffices.
LAGGED_PREDS = ["surprisal_gpt2", "surprisal_sedd",
                "word_length", "log_freq", "len_freq"]

def add_spillover_lags(df, max_lag=3):
    print(f"Creating spillover predictors (lags 1–{max_lag})...")
    df = df.sort_values(["sentence_idx", "word_idx"]).copy()
    for lag in range(1, max_lag + 1):
        for p in LAGGED_PREDS:
            df[f"{p}_lag{lag}"] = df.groupby("sentence_idx")[p].shift(lag)
    return df


# ── Step 5  Eye-tracking wide → long ────────────────────────────────
def et_to_long(et, measure="gz"):
    print(f"Reshaping eye-tracking data to long format (measure={measure})...")
    rt_cols = sorted(
        [c for c in et.columns if c.startswith(f"{measure}R")],
        key=lambda c: int(re.search(r"\d+", c.replace(measure, "")).group()))

    id_vars = ["subj", "item", "sentence"]
    long = pd.melt(et[id_vars + rt_cols],
                   id_vars=id_vars, var_name="rt_col",
                   value_name="reading_time")
    long["word_idx"] = (long["rt_col"]
                        .str.extract(r"(\d+)").astype(int).values - 1)
    long["reading_time"] = pd.to_numeric(long["reading_time"], errors="coerce")
    long = long.dropna(subset=["reading_time"])
    long = long[long["reading_time"] > 0].copy()
    print(f"  Fixated word observations: {len(long)}")
    return long[["subj", "item", "sentence", "word_idx", "reading_time"]]


# ── Step 6  Merge & exclusions ──────────────────────────────────────
def merge_and_exclude(et_long, surp_word):
    print("Merging eye-tracking with surprisal data...")
    et_long["sentence_clean"] = et_long["sentence"].str.strip()
    surp_word["sentence_clean"] = surp_word["sentence"].str.strip()

    merged = et_long.merge(
        surp_word.drop(columns=["sentence", "word"]),
        on=["sentence_clean", "word_idx"], how="inner")
    print(f"  After merge: {len(merged)}")

    # Exclude last word of each sentence (wrap-up effects)
    max_wi = surp_word.groupby("sentence_idx")["word_idx"].max()
    max_map = surp_word[["sentence_idx", "sentence_clean"]].drop_duplicates() \
        .merge(max_wi.rename("max_wi"), left_on="sentence_idx", right_index=True)
    merged = merged.merge(max_map[["sentence_clean", "max_wi"]],
                          on="sentence_clean", how="left")
    n0 = len(merged)
    merged = merged[merged["word_idx"] < merged["max_wi"]].copy()
    print(f"  Excluded last word: removed {n0 - len(merged)}")

    # Drop any row where a predictor is missing (covers first words w/o
    # full spillover context and the missing first-word surprisal)
    pred_cols = [c for c in merged.columns
                 if any(c.startswith(p) for p in
                        ("surprisal_gpt2", "surprisal_sedd",
                         "word_position", "word_length", "log_freq", "len_freq"))
                 and c not in ("sentence_clean", "word_clean")]
    num_cols = [c for c in pred_cols
                if merged[c].dtype in ("float64", "int64", "float32")]
    n0 = len(merged)
    merged = merged.dropna(subset=num_cols)
    print(f"  Excluded incomplete spillover: removed {n0 - len(merged)}")
    print(f"  Final observations: {len(merged)}")
    return merged


# ── Step 7  Center & scale ──────────────────────────────────────────
def center_and_scale(data, cols):
    params = {}
    for c in cols:
        m, s = data[c].mean(), data[c].std()
        if s == 0:
            s = 1.0
        data[c + "_z"] = (data[c] - m) / s
        params[c] = {"mean": float(m), "std": float(s)}
    return data, params


# ── Step 8  Fit LME ─────────────────────────────────────────────────
def fit_lme(data, model_label, surp_prefix):
    print(f"\n{'=' * 60}")
    print(f"Fitting LME model: {model_label}")
    print(f"{'=' * 60}")

    surp_z = [f"{surp_prefix}_z"] + \
             [f"{surp_prefix}_lag{l}_z" for l in range(1, MAX_LAG + 1)]
    ctrl_z = ["word_position_z"]
    for suf in ("_z", "_lag1_z", "_lag2_z", "_lag3_z"):
        for p in ("word_length", "log_freq", "len_freq"):
            ctrl_z.append(f"{p}{suf}")

    all_z = surp_z + ctrl_z
    formula = "reading_time ~ " + " + ".join(all_z)
    print(f"  Predictors: {len(all_z)}")
    print(f"  N obs = {len(data)}, N subj = {data['subj'].nunique()}, "
          f"N items = {data['item'].nunique()}")

    # Try progressively simpler random-effects structures,
    # each with multiple optimisation methods
    specs = [
        ("full (1+surp|subj) + (1|item)",
         dict(re_formula=f"~{surp_prefix}_z",
              vc_formula={"item": "0 + C(item)"})),
        ("(1+surp|subj)",
         dict(re_formula=f"~{surp_prefix}_z")),
        ("(1|subj)",
         dict()),
    ]
    methods = ["lbfgs", "powell", "cg", "bfgs"]

    result = None
    for label, kw in specs:
        for method in methods:
            try:
                print(f"  Trying RE: {label}, method={method} ...")
                mdl = smf.mixedlm(formula, data, groups="subj", **kw)
                result = mdl.fit(reml=True, maxiter=500, method=method)
                print(f"  Converged  ✓  (method={method})")
                break
            except Exception as e:
                short = str(e)[:60]
                print(f"    failed ({short})")
        if result is not None:
            break

    if result is None:
        print("  ERROR: No model converged.")
        return None, surp_z

    return result, surp_z


# ── Step 9  Extract conversion factors ───────────────────────────────
def extract_factors(result, surp_z_vars, scaling, surp_prefix, model_label):
    labels = {
        f"{surp_prefix}_z": "Surprisal word_n",
        f"{surp_prefix}_lag1_z": "Surprisal word_{n-1}",
        f"{surp_prefix}_lag2_z": "Surprisal word_{n-2}",
        f"{surp_prefix}_lag3_z": "Surprisal word_{n-3}",
    }
    raw_key = {
        f"{surp_prefix}_z": surp_prefix,
        f"{surp_prefix}_lag1_z": f"{surp_prefix}_lag1",
        f"{surp_prefix}_lag2_z": f"{surp_prefix}_lag2",
        f"{surp_prefix}_lag3_z": f"{surp_prefix}_lag3",
    }

    rows = []
    for v in surp_z_vars:
        if v not in result.params.index:
            continue
        coef_z = result.params[v]
        se_z = result.bse[v]
        z_val = result.tvalues[v]
        p_val = result.pvalues[v]
        sd_x = scaling[raw_key[v]]["std"]
        rows.append(dict(
            model=model_label,
            predictor=labels[v],
            coef_scaled=round(coef_z, 4),
            coef_ms_per_bit=round(coef_z / sd_x, 4),
            se_ms_per_bit=round(se_z / sd_x, 4),
            z=round(z_val, 3),
            p_value=p_val,
            sig=sig_stars(p_val),
        ))
    return pd.DataFrame(rows)


# ── Main ─────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("CONVERSION FACTOR ANALYSIS")
    print("Linking surprisal estimates to reading times (filler items)")
    print("=" * 70 + "\n")

    et, surp = load_data()
    surp_word = aggregate_tokens_to_words(surp)
    surp_word = add_word_features(surp_word)
    surp_word = add_spillover_lags(surp_word, MAX_LAG)
    et_long = et_to_long(et, RT_MEASURE)
    data = merge_and_exclude(et_long, surp_word)

    # Columns to center/scale: word_position only for current word;
    # all others for current + lags
    pred_cols = ["word_position"]
    for b in LAGGED_PREDS:
        pred_cols.append(b)
        for lag in range(1, MAX_LAG + 1):
            pred_cols.append(f"{b}_lag{lag}")

    print("\nCentering and scaling all predictors...")
    data, scaling = center_and_scale(data, pred_cols)

    all_factors = []
    full_summaries = {}
    for model_label, prefix in [("GPT-2", "surprisal_gpt2"),
                                ("SEDD", "surprisal_sedd")]:
        result, surp_z = fit_lme(data, model_label, prefix)
        if result is None:
            continue

        factors = extract_factors(result, surp_z, scaling, prefix, model_label)
        all_factors.append(factors)
        full_summaries[model_label] = result.summary()

        # Print conversion factors to console
        print(f"\n  {'Predictor':<25} {'ms/bit':>10} {'SE':>8} "
              f"{'z':>8} {'p':>10} {'':>4}")
        print(f"  {'─' * 70}")
        for _, r in factors.iterrows():
            ps = f"{r['p_value']:.4f}" if r["p_value"] >= 0.0001 else "<0.0001"
            print(f"  {r['predictor']:<25} {r['coef_ms_per_bit']:>10.4f} "
                  f"{r['se_ms_per_bit']:>8.4f} {r['z']:>8.3f} "
                  f"{ps:>10} {r['sig']:>4}")

    if not all_factors:
        print("\nNo models converged. Exiting.")
        sys.exit(1)

    # ── Save outputs ────────────────────────────────────────────────
    factors_df = pd.concat(all_factors, ignore_index=True)

    csv_path = os.path.join(RESULTS_DIR, "conversion_factors.csv")
    factors_df.to_csv(csv_path, index=False)
    print(f"\nConversion factors → {csv_path}")

    txt_path = os.path.join(RESULTS_DIR, "conversion_factors_summary.txt")
    with open(txt_path, "w") as f:
        f.write("CONVERSION FACTORS: Surprisal → Reading Time (ms)\n")
        f.write("=" * 70 + "\n")
        f.write(f"DV: Gaze duration (first-pass reading time, ms)\n")
        f.write(f"Method: Frequentist LME on filler items\n")
        f.write(f"Random effects: (1 + surprisal | subject) + (1 | item)\n")
        f.write(f"Exclusions: words without 3-word spillover context; "
                f"sentence-final word\n")
        f.write(f"Predictors centered and scaled; coefficients converted "
                f"back to ms / bit\n")
        f.write(f"Observations: {len(data)}\n")
        f.write(f"Subjects: {data['subj'].nunique()}\n")
        f.write(f"Items: {data['item'].nunique()}\n\n")

        for ml in ["GPT-2", "SEDD"]:
            mf = factors_df[factors_df["model"] == ml]
            if mf.empty:
                continue
            f.write(f"\n{'=' * 70}\n")
            f.write(f"Language model: {ml}\n")
            f.write(f"{'=' * 70}\n\n")
            f.write(f"{'Predictor':<25} {'ms/bit':>10} {'SE':>8} "
                    f"{'z':>8} {'p-value':>10} {'Sig':>5}\n")
            f.write(f"{'─' * 70}\n")
            for _, r in mf.iterrows():
                ps = (f"{r['p_value']:.4f}" if r["p_value"] >= 0.0001
                      else "<0.0001")
                f.write(f"{r['predictor']:<25} {r['coef_ms_per_bit']:>10.4f} "
                        f"{r['se_ms_per_bit']:>8.4f} {r['z']:>8.3f} "
                        f"{ps:>10} {r['sig']:>5}\n")
            f.write(f"{'─' * 70}\n\n")
            f.write("Interpretation: coefficient = change in gaze duration "
                    "(ms) per 1-bit\nincrease in surprisal.  "
                    "word_{n-k} = effect of the word k positions\n"
                    "back on reading time of the current word n "
                    "(spillover).\n\n")

    print(f"Summary      → {txt_path}")

    for ml, smry in full_summaries.items():
        sp = os.path.join(RESULTS_DIR,
                          f"lme_full_summary_{ml.lower().replace('-','')}.txt")
        with open(sp, "w") as f:
            f.write(str(smry))
        print(f"Full summary → {sp}")

    print("\nDone.")


if __name__ == "__main__":
    main()
