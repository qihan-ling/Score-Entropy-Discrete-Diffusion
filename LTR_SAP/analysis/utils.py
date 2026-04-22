"""
Shared utilities for LTR-SAP analysis.

Provides data loading, metric I/O, token-to-word alignment, and plotting helpers
used across all analysis scripts (Plans A, B, C).
"""

import json
import os
import csv
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path constants (relative to repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SAP_STIMULI_DIR = REPO_ROOT / "SAP_stimuli"
LTR_SAP_DIR = REPO_ROOT / "LTR_SAP"
LTR_SAP_CRITICAL_DIR = REPO_ROOT / "LTR_SAP_critical"
ET_DATA_PATH = (
    REPO_ROOT
    / "Huang_et_al_2024_spr_osf"
    / "material & exp_script"
    / "EM_analysis"
    / "R"
    / "all_wide.csv"
)
FILLER_ET_PATH = (
    REPO_ROOT
    / "Huang_et_al_2024_spr_osf"
    / "material & exp_script"
    / "EM_analysis"
    / "R"
    / "filler_wide.csv"
)
SPR_DIR = REPO_ROOT / "sapbenchmark" / "analysis" / "spr"
GPT2_SURP_DIR = REPO_ROOT / "sapbenchmark" / "Surprisals" / "data" / "gpt2"
POSITION_INFO_PATH = (
    REPO_ROOT
    / "Huang_et_al_2024_spr_osf"
    / "material & exp_script"
    / "EM_analysis"
    / "make_cnt"
    / "Position_Info.csv"
)

# Mapping from CSV filename stems to their critical-position column name
CRITICAL_POS_COLUMN = {
    "sap_items_Agreement": "disambPosition",
    "sap_items_ClassicGP": "disambPosition",
    "sap_items_AttachmentAmbiguity": "disambPosition",
    "sap_items_RelativeClause": "targetPosition",
    "sap_items_filler": None,
}

CONDITION_COLUMN = {
    "sap_items_Agreement": "condition",
    "sap_items_ClassicGP": "condition",
    "sap_items_AttachmentAmbiguity": "condition",
    "sap_items_RelativeClause": "condition",
    "sap_items_filler": None,
}

ET_MEASURES = ["ffd", "gz", "gp", "tt", "regin", "regout"]
SPR_MEASURES = ["RT"]


# ---------------------------------------------------------------------------
# SAP stimuli loading
# ---------------------------------------------------------------------------

def load_sap_csv(csv_path):
    """Load a SAP stimuli CSV into a pandas DataFrame."""
    return pd.read_csv(csv_path)


def get_sap_files():
    """Return sorted list of SAP CSV paths."""
    return sorted(SAP_STIMULI_DIR.glob("sap_items_*.csv"))


def get_subset_name(csv_path):
    """Extract subset name from CSV path (e.g., 'Agreement' from 'sap_items_Agreement.csv')."""
    stem = Path(csv_path).stem
    return stem.replace("sap_items_", "")


def get_critical_pos_col(csv_path):
    """Get the critical position column name for this CSV."""
    stem = Path(csv_path).stem
    return CRITICAL_POS_COLUMN.get(stem)


def get_condition_col(csv_path):
    """Get the condition column name for this CSV."""
    stem = Path(csv_path).stem
    return CONDITION_COLUMN.get(stem)


def filter_stimuli(df, condition=None, ambiguous=None):
    """Filter a SAP stimuli DataFrame by condition and/or ambiguous flag.

    Used by the batch runners to support per-condition sbatch parallelism via
    --condition and --ambiguous CLI flags. Either/both filters can be None to
    skip. `ambiguous` is ignored if the CSV has no 'ambiguous' column, so
    passing `--ambiguous 0` to a non-ClassicGP subset is a harmless no-op.

    Args:
        df: DataFrame loaded via load_sap_csv.
        condition: optional str; matched against the 'condition' column.
        ambiguous: optional int (0 or 1); matched against the 'ambiguous' column.

    Returns:
        Filtered DataFrame (not copied).
    """
    if condition is not None and "condition" in df.columns:
        df = df[df["condition"] == condition]
    if ambiguous is not None and "ambiguous" in df.columns:
        df = df[df["ambiguous"].astype(int) == int(ambiguous)]
    return df


def iter_sap_items(csv_path):
    """Iterate over (item_number, condition, disambPos, sentence) from a SAP CSV.

    Yields:
        dict with keys: item, condition, critical_pos, sentence, csv_stem
    """
    df = load_sap_csv(csv_path)
    stem = Path(csv_path).stem
    subset = get_subset_name(csv_path)
    crit_col = get_critical_pos_col(csv_path)
    cond_col = get_condition_col(csv_path)

    for _, row in df.iterrows():
        yield {
            "item": row.get("item", None),
            "condition": row[cond_col] if cond_col and cond_col in row else None,
            "critical_pos": int(row[crit_col]) if crit_col and crit_col in row else None,
            "sentence": row["Sentence"],
            "subset": subset,
        }


# ---------------------------------------------------------------------------
# SEDD output I/O
# ---------------------------------------------------------------------------

def load_sedd_output(json_path):
    """Load a SEDD enforce-prefix output JSON."""
    with open(json_path) as f:
        return json.load(f)


def save_sedd_output(data, json_path):
    """Save SEDD output to JSON."""
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def get_output_path(subset, condition, item):
    """Get the output JSON path for a specific item.

    Returns e.g. LTR_SAP/Agreement/AGREE/item_1.json
    """
    if condition:
        return LTR_SAP_DIR / subset / condition / f"item_{item}.json"
    return LTR_SAP_DIR / subset / f"item_{item}.json"


def load_all_outputs(subset, condition=None):
    """Load all SEDD output JSONs for a subset/condition.

    Returns:
        list of dicts (loaded JSONs)
    """
    if condition:
        pattern_dir = LTR_SAP_DIR / subset / condition
    else:
        pattern_dir = LTR_SAP_DIR / subset
    results = []
    if pattern_dir.exists():
        for f in sorted(pattern_dir.glob("item_*.json")):
            results.append(load_sedd_output(f))
    return results


# ---------------------------------------------------------------------------
# GPT-2 surprisal loading (from sapbenchmark)
# ---------------------------------------------------------------------------

def load_gpt2_surprisals(subset_name):
    """Load pre-computed GPT-2 word-level surprisals from sapbenchmark.

    Maps subset names to filenames:
        Agreement -> items_Agreement.gpt2.csv.scaled
        ClassicGP -> items_ClassicGP.gpt2.csv.scaled
        etc.
    """
    fname = f"items_{subset_name}.gpt2.csv.scaled"
    path = GPT2_SURP_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"GPT-2 surprisals not found: {path}")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Human behavioral data loading
# ---------------------------------------------------------------------------

def load_eye_tracking(path=None):
    """Load eye-tracking data from all_wide.csv."""
    path = path or ET_DATA_PATH
    return pd.read_csv(path)


def load_filler_eye_tracking(path=None):
    """Load filler eye-tracking data from filler_wide.csv."""
    path = path or FILLER_ET_PATH
    return pd.read_csv(path)


def load_spr_data(subset_name):
    """Load SPR data for a subset.

    Maps subset names to CSV filenames in sapbenchmark/analysis/spr/:
        Fillers -> Fillers.csv
        Agreement -> AgreementSet.csv
        ClassicGP -> ClassicGardenPathSet.csv
        RelativeClause -> RelativeClauseSet.csv
        AttachmentAmbiguity -> AttachmentSet.csv
    """
    name_map = {
        "Fillers": "Fillers.csv",
        "filler": "Fillers.csv",
        "Agreement": "AgreementSet.csv",
        "ClassicGP": "ClassicGardenPathSet.csv",
        "RelativeClause": "RelativeClauseSet.csv",
        "AttachmentAmbiguity": "AttachmentSet.csv",
    }
    fname = name_map.get(subset_name)
    if not fname:
        raise ValueError(f"Unknown subset: {subset_name}")
    path = SPR_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"SPR data not found: {path}")
    df = pd.read_csv(path)
    if "MD5" in df.columns:
        df["participant"] = df["MD5"]
    return df


def load_position_info():
    """Load Position_Info.csv mapping (item, cond) -> ROI positions."""
    return pd.read_csv(POSITION_INFO_PATH)


# ---------------------------------------------------------------------------
# Token-to-word alignment
# ---------------------------------------------------------------------------

def align_commitment_log_to_words(commitment_log, tokenization_info):
    """Align token-level commitment_log entries to word positions.

    Uses the sentence text and token IDs to map each token position to the
    corresponding whitespace-delimited word index.

    Args:
        commitment_log: list of dicts from SEDD output
        tokenization_info: dict with 'sentence', 'full_ids'

    Returns:
        list of dicts, each augmented with 'word_pos' (0-indexed)
    """
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", add_prefix_space=True)

    sentence = tokenization_info["sentence"]
    words = sentence.split()
    tokens = tokenizer.tokenize(sentence)

    # Build alignment using the sapbenchmark method
    def _clean(token):
        return re.sub(r"[^a-zA-Z0-9*.,!?\-]", "", token)

    cleaned = [_clean(t) for t in tokens]
    aligned = []
    idx_word = 0
    current_pieces = []

    for idx_piece, piece in enumerate(cleaned):
        if idx_word < len(words):
            word = words[idx_word]
        else:
            current_pieces += cleaned[idx_piece:]
            break
        if piece == word[:len(piece)]:
            aligned.append(current_pieces)
            idx_word += 1
            current_pieces = [piece]
        else:
            current_pieces.append(piece)
    aligned.append(current_pieces)
    aligned = aligned[1:]

    breaks = [len(pieces) for pieces in aligned]
    breaks = [0] + [sum(breaks[:i + 1]) for i in range(len(breaks))]

    # Map token index (0-indexed within sentence tokens, i.e., position - 1 since
    # position 0 is <|endoftext|>) to word index
    token_to_word = {}
    for word_idx in range(len(breaks) - 1):
        for tok_idx in range(breaks[word_idx], breaks[word_idx + 1]):
            token_to_word[tok_idx] = word_idx

    result = []
    for entry in commitment_log:
        e = dict(entry)
        tok_idx = entry["position"] - 1  # subtract 1 for <|endoftext|>
        e["word_pos"] = token_to_word.get(tok_idx)
        result.append(e)
    return result


def aggregate_metrics_by_word(aligned_log, merge_fn="sum"):
    """Aggregate token-level metrics to word level.

    For multi-token words, steps_to_commit is summed and surprisal is summed
    (matching sapbenchmark's sum_surprisal convention).

    Args:
        aligned_log: list of dicts with 'word_pos' from align_commitment_log_to_words
        merge_fn: "sum" or "mean"

    Returns:
        list of dicts, one per word position
    """
    by_word = defaultdict(list)
    for entry in aligned_log:
        wp = entry.get("word_pos")
        if wp is not None:
            by_word[wp].append(entry)

    fn = sum if merge_fn == "sum" else lambda xs: sum(xs) / len(xs)
    result = []
    for wp in sorted(by_word.keys()):
        entries = by_word[wp]
        agg = {
            "word_pos": wp,
            "steps_to_commit": fn([e["steps_taken"] for e in entries]),
            "n_tokens": len(entries),
        }
        surprisals = [e["final_surprisal"] for e in entries if e.get("final_surprisal") is not None]
        if surprisals:
            agg["surprisal"] = fn(surprisals)
        entropies = [e["final_entropy"] for e in entries if e.get("final_entropy") is not None]
        if entropies:
            agg["entropy"] = fn(entropies)
        kls = [e["cumulative_kl"] for e in entries if e.get("cumulative_kl") is not None]
        if kls:
            agg["cumulative_kl"] = fn(kls)
        result.append(agg)
    return result


# ---------------------------------------------------------------------------
# Weighted steps metric (for strict-LTR position deconfounding)
# ---------------------------------------------------------------------------

def compute_weighted_steps(commitment_log, total_steps=1024):
    """Compute weighted_steps = steps_taken * (1 - t_commitment).

    t_commitment = 1 - step/total_steps is the noise level at commitment.
    At high noise (t near 1), the weight (1-t) is small; at low noise the weight
    is large. This normalizes for the position confound where later positions
    commit at lower noise and take fewer steps.

    Args:
        commitment_log: list of dicts from SEDD strict-LTR output
        total_steps: total number of denoising steps used

    Returns:
        list of dicts, each augmented with 'weighted_steps' and 't_commitment'
    """
    result = []
    for entry in commitment_log:
        e = dict(entry)
        step = entry["step"]
        t_commit = 1.0 - step / total_steps
        e["t_commitment"] = t_commit
        e["weighted_steps"] = entry["steps_taken"] * (1.0 - t_commit)
        result.append(e)
    return result


# ---------------------------------------------------------------------------
# Critical-position output loading
# ---------------------------------------------------------------------------

def load_critical_output(json_path):
    """Load a critical-position experiment output JSON."""
    with open(json_path) as f:
        return json.load(f)


def load_all_critical_outputs(subset, condition=None):
    """Load all critical-position outputs for a subset/condition.

    Returns:
        list of dicts (loaded JSONs)
    """
    if condition:
        pattern_dir = LTR_SAP_CRITICAL_DIR / subset / condition
    else:
        pattern_dir = LTR_SAP_CRITICAL_DIR / subset
    results = []
    if pattern_dir.exists():
        for f in sorted(pattern_dir.glob("item_*_pos_*.json")):
            results.append(load_critical_output(f))
    return results


def load_critical_outputs_by_offset(subset, condition, offsets=None):
    """Load critical-position outputs grouped by offset.

    Args:
        subset: e.g. "Agreement"
        condition: e.g. "AGREE"
        offsets: list of offsets to load, default [-2, -1, 0, 1, 2, 3]

    Returns:
        dict: offset -> list of loaded JSONs
    """
    if offsets is None:
        offsets = [-2, -1, 0, 1, 2, 3]
    result = {}
    if condition:
        d = LTR_SAP_CRITICAL_DIR / subset / condition
    else:
        d = LTR_SAP_CRITICAL_DIR / subset
    for offset in offsets:
        sign = "+" if offset >= 0 else ""
        pattern = f"item_*_pos_{sign}{offset}.json"
        files = sorted(d.glob(pattern)) if d.exists() else []
        result[offset] = [load_critical_output(f) for f in files]
    return result


# ---------------------------------------------------------------------------
# Extracting metrics from SEDD output for analysis
# ---------------------------------------------------------------------------

def extract_word_metrics(sedd_output):
    """Extract word-level metrics from a single SEDD output JSON.

    Returns a DataFrame with columns:
        word_pos, word, steps_to_commit, surprisal, entropy, cumulative_kl, n_tokens
    """
    commitment_log = sedd_output["commitment_log"]
    tokenization = sedd_output["tokenization"]

    aligned = align_commitment_log_to_words(commitment_log, tokenization)
    word_agg = aggregate_metrics_by_word(aligned, merge_fn="sum")

    sentence = tokenization["sentence"]
    words = sentence.split()
    for entry in word_agg:
        wp = entry["word_pos"]
        if wp < len(words):
            entry["word"] = words[wp]
    return pd.DataFrame(word_agg)


def extract_critical_region_metrics(sedd_output, critical_pos, window=3):
    """Extract metrics for positions [critical_pos - window, critical_pos + window].

    Args:
        sedd_output: loaded SEDD output JSON
        critical_pos: 1-indexed word position of the critical/disamb/target word
        window: number of words before/after to include

    Returns:
        DataFrame with columns: word_pos, relative_pos, word, steps_to_commit, ...
    """
    word_df = extract_word_metrics(sedd_output)
    crit_0indexed = critical_pos - 1  # SAP CSVs use 1-indexed positions

    word_df["relative_pos"] = word_df["word_pos"] - crit_0indexed
    mask = (word_df["relative_pos"] >= -window) & (word_df["relative_pos"] <= window)
    return word_df[mask].copy()


# ---------------------------------------------------------------------------
# Entropy trajectory extraction
# ---------------------------------------------------------------------------

def extract_entropy_trajectory(sedd_output, position):
    """Extract the entropy trajectory (list of entropy values over denoising steps)
    for a specific token position.

    Args:
        sedd_output: loaded SEDD output JSON
        position: token position (int)

    Returns:
        list of floats (entropy at each step), or empty list if not tracked
    """
    hist = sedd_output.get("frontier_history", {}).get(str(position), [])
    return [h["entropy"] for h in hist]


def extract_all_trajectories(sedd_output, metric="entropy"):
    """Extract trajectory for all tracked positions.

    Returns:
        dict: position (int) -> list of metric values over steps
    """
    result = {}
    for pos_str, hist in sedd_output.get("frontier_history", {}).items():
        result[int(pos_str)] = [h.get(metric, 0) for h in hist]
    return result


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def setup_matplotlib():
    """Configure matplotlib for publication-quality figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })
    return plt


def condition_palette():
    """Return a color palette for SAP conditions."""
    return {
        "AGREE": "#2196F3",
        "UNAGREE": "#F44336",
        "NPS_AMB": "#FF9800",
        "NPS_UAMB": "#4CAF50",
        "NPZ_AMB": "#FF9800",
        "NPZ_UAMB": "#4CAF50",
        "MVRR_AMB": "#FF9800",
        "MVRR_UAMB": "#4CAF50",
        "RC_Subj": "#2196F3",
        "RC_Obj": "#F44336",
        "AttachMulti": "#9C27B0",
        "AttachHigh": "#FF9800",
        "AttachLow": "#4CAF50",
    }
