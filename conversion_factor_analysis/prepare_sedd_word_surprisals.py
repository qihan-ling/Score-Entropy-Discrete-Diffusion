#!/usr/bin/env python3
"""
Aggregate BPE-token-level SEDD surprisals to word-level,
matching the methodology of sapbenchmark/Surprisals/get_gpt2_full.py.

Input:  outputs/sedd_gpt2_sap_filler_merged.csv  (token-level)
Output: conversion_factor_analysis/sedd_filler_word.csv  (word-level)

The output mirrors the structure of items_filler.gpt2.csv so that
rescale.py-style processing and Fillers_analysis.R can consume it.
"""

import os
import sys
import csv
import re
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import clean, align

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKEN_CSV = os.path.join(BASE_DIR, "outputs", "sedd_gpt2_sap_filler_merged.csv")
ITEMS_CSV = os.path.join(BASE_DIR, "sapbenchmark", "Surprisals", "data", "items_filler.pivot.csv")
FREQS_CSV = os.path.join(BASE_DIR, "sapbenchmark", "Surprisals", "analysis", "freqs_coca.csv")
OUT_CSV   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sedd_filler_word.csv")


def load_coca_freqs(path):
    freqs = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            freqs[row["word"]] = int(row["count"])
    return freqs


def build_item_map(items_path):
    """Map sentence text → item number from items_filler.pivot.csv."""
    items = pd.read_csv(items_path)
    return dict(zip(items["Sentence"], items["item"]))


def aggregate_tokens_to_words(token_df):
    """Aggregate BPE token surprisals → word-level using utils.align().

    The token CSV skips the first word of each sentence, so we align
    against words[1:] (all words except the first).
    """
    rows = []
    for sidx, grp in token_df.groupby("sentence_idx"):
        sentence = grp["sentence"].iloc[0]
        words_all = sentence.split()
        grp = grp.sort_values("word_position")

        tokens_raw = grp["word"].tolist()
        tokens_clean = [clean(t) for t in tokens_raw]
        surp_sedd = grp["surprisal_sedd"].values

        target_words = words_all[1:]
        try:
            _, breaks = align(target_words, tokens_clean)
        except Exception:
            breaks = _space_breaks(tokens_raw)

        # word 0 (first word) has no surprisal — skip it
        for i, word in enumerate(target_words):
            lo, hi = breaks[i], breaks[i + 1]
            rows.append({
                "sentence_idx": sidx,
                "Sentence": sentence,
                "word": word,
                "word_pos": i + 1,           # 0-indexed matching sapbenchmark (word 0 = first word)
                "sum_surprisal": float(surp_sedd[lo:hi].sum()),
                "mean_surprisal": float(surp_sedd[lo:hi].mean()),
            })

    return pd.DataFrame(rows)


def _space_breaks(tokens_raw):
    breaks = [0]
    for i, t in enumerate(tokens_raw):
        if t.startswith(" ") and i > 0:
            breaks.append(i)
    breaks.append(len(tokens_raw))
    return breaks


def normalize_sentence(s):
    """Undo GPT-2 tokenizer spacing artifacts for sentence matching."""
    s = re.sub(r"\s+'s\b", "'s", s)
    s = re.sub(r"\s+n't\b", "n't", s)
    s = re.sub(r"\s+'t\b", "'t", s)
    s = re.sub(r"\s+'re\b", "'re", s)
    s = re.sub(r"\s+'ve\b", "'ve", s)
    s = re.sub(r"\s+'ll\b", "'ll", s)
    s = re.sub(r"\s+'d\b", "'d", s)
    s = re.sub(r"\s+'m\b", "'m", s)
    return s


def add_features(df, coca_freqs):
    """Add item, logfreq, length — matching rescale.py methodology."""
    item_map = build_item_map(ITEMS_CSV)
    df["Sentence_norm"] = df["Sentence"].apply(normalize_sentence)
    df["item"] = df["Sentence_norm"].map(item_map)
    df.drop(columns=["Sentence_norm"], inplace=True)

    def get_logfreq(word):
        w = re.sub("[.,?!;:]", "", word.lower())
        count = coca_freqs.get(w, 0)
        return np.log(count) if count > 0 else np.nan

    df["logfreq"] = df["word"].apply(get_logfreq)
    df["length"] = df["word"].apply(len)

    return df


def main():
    print("=== Preparing SEDD word-level surprisals ===\n")

    print(f"Reading token-level data: {TOKEN_CSV}")
    tokens = pd.read_csv(TOKEN_CSV)
    print(f"  {len(tokens)} token rows, {tokens['sentence_idx'].nunique()} sentences\n")

    print("Aggregating BPE tokens → words...")
    words = aggregate_tokens_to_words(tokens)
    print(f"  {len(words)} word-level rows\n")

    print(f"Loading COCA frequencies: {FREQS_CSV}")
    coca = load_coca_freqs(FREQS_CSV)
    print(f"  {len(coca)} words in COCA\n")

    print("Adding features (item, logfreq, length)...")
    words = add_features(words, coca)

    missing_items = words["item"].isna().sum()
    if missing_items > 0:
        print(f"  WARNING: {missing_items} rows could not be mapped to items")

    print(f"\nSaving to: {OUT_CSV}")
    words.to_csv(OUT_CSV, index=False)

    print(f"\n=== Summary ===")
    print(f"  Total words: {len(words)}")
    print(f"  Unique sentences: {words['sentence_idx'].nunique()}")
    print(f"  Unique items: {words['item'].nunique()}")
    print(f"  sum_surprisal range: [{words['sum_surprisal'].min():.3f}, {words['sum_surprisal'].max():.3f}]")
    print(f"  sum_surprisal mean: {words['sum_surprisal'].mean():.3f}")
    print(f"  sum_surprisal SD: {words['sum_surprisal'].std():.3f}")
    print(f"  logfreq coverage: {words['logfreq'].notna().sum()}/{len(words)}")


if __name__ == "__main__":
    main()
