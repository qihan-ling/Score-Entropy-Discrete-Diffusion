"""
Utilities for grouping GPT-2 vocabulary tokens by syntactic (POS) and
semantic (embedding-neighbor) categories.

Used by --track_token_groups in the critical-position / critical-region /
bidirectional scripts.

Usage:
    from token_group_utils import TokenGroupTracker
    tracker = TokenGroupTracker(tokenizer, model, device)
    metrics = tracker.compute_group_metrics(probs_tensor, target_token_id)
"""

import json
import os
from pathlib import Path

import torch
import numpy as np

CACHE_DIR = Path(__file__).resolve().parent / ".token_caches"

# Coarse POS categories
POS_CATEGORIES = ["NOUN", "VERB", "ADJ", "ADV", "DET", "PREP", "CONJ", "PRON", "PUNCT", "NUM", "OTHER"]

# NLTK universal POS tag mapping to our coarse categories
_NLTK_TO_COARSE = {
    "NN": "NOUN", "NNS": "NOUN", "NNP": "NOUN", "NNPS": "NOUN",
    "VB": "VERB", "VBD": "VERB", "VBG": "VERB", "VBN": "VERB", "VBP": "VERB", "VBZ": "VERB",
    "MD": "VERB",
    "JJ": "ADJECTIVE", "JJR": "ADJECTIVE", "JJS": "ADJECTIVE",
    "RB": "ADVERB", "RBR": "ADVERB", "RBS": "ADVERB",
    "DT": "DET", "PDT": "DET", "WDT": "DET",
    "IN": "PREP", "TO": "PREP",
    "CC": "CONJ",
    "PRP": "PRON", "PRP$": "PRON", "WP": "PRON", "WP$": "PRON",
    ".": "PUNCT", ",": "PUNCT", ":": "PUNCT", "``": "PUNCT", "''": "PUNCT",
    "-LRB-": "PUNCT", "-RRB-": "PUNCT",
    "CD": "NUM",
}


def _build_pos_cache(tokenizer, cache_path):
    """Build a token_id -> POS_category mapping using NLTK."""
    try:
        import nltk
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    except ImportError:
        print("WARNING: nltk not available, falling back to heuristic POS tagging")
        return _build_pos_cache_heuristic(tokenizer, cache_path)

    vocab = tokenizer.get_vocab()
    pos_map = {}

    # Decode each token, clean it, and tag
    tokens_to_tag = []
    id_list = []
    for token_str, token_id in vocab.items():
        decoded = tokenizer.decode([token_id]).strip()
        if decoded and decoded.isalpha() and len(decoded) > 1:
            tokens_to_tag.append(decoded.lower())
            id_list.append(token_id)

    # Batch tag
    if tokens_to_tag:
        tagged = nltk.pos_tag(tokens_to_tag)
        for (word, tag), token_id in zip(tagged, id_list):
            coarse = _NLTK_TO_COARSE.get(tag, "OTHER")
            pos_map[token_id] = coarse

    # Fill remaining tokens with OTHER
    for token_id in range(len(vocab)):
        if token_id not in pos_map:
            pos_map[token_id] = "OTHER"

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({str(k): v for k, v in pos_map.items()}, f)

    return pos_map


def _build_pos_cache_heuristic(tokenizer, cache_path):
    """Heuristic POS tagging based on common suffixes."""
    vocab = tokenizer.get_vocab()
    pos_map = {}

    suffix_rules = [
        (("tion", "sion", "ment", "ness", "ity", "ence", "ance"), "NOUN"),
        (("ing", "ed", "ize", "ise", "ate"), "VERB"),
        (("ous", "ful", "less", "able", "ible", "ive", "al", "ical"), "ADJECTIVE"),
        (("ly",), "ADVERB"),
    ]

    det_words = {"the", "a", "an", "this", "that", "these", "those", "my", "your", "his",
                 "her", "its", "our", "their", "some", "any", "no", "every", "each"}
    prep_words = {"in", "on", "at", "to", "for", "with", "by", "from", "of", "about",
                  "into", "through", "during", "before", "after", "above", "below",
                  "between", "under", "over"}
    conj_words = {"and", "but", "or", "nor", "for", "yet", "so", "because", "although",
                  "while", "if", "when", "that", "which", "who", "whom", "whose"}
    pron_words = {"i", "me", "my", "mine", "we", "us", "our", "ours", "you", "your",
                  "yours", "he", "him", "his", "she", "her", "hers", "it", "its",
                  "they", "them", "their", "theirs", "who", "whom", "what", "which"}

    for token_str, token_id in vocab.items():
        decoded = tokenizer.decode([token_id]).strip().lower()
        if not decoded:
            pos_map[token_id] = "OTHER"
            continue

        if decoded in det_words:
            pos_map[token_id] = "DET"
        elif decoded in prep_words:
            pos_map[token_id] = "PREP"
        elif decoded in conj_words:
            pos_map[token_id] = "CONJ"
        elif decoded in pron_words:
            pos_map[token_id] = "PRON"
        elif not decoded.isalpha():
            pos_map[token_id] = "PUNCT" if any(c in decoded for c in ".,;:!?-()[]{}") else "OTHER"
        elif decoded.isdigit():
            pos_map[token_id] = "NUM"
        else:
            matched = False
            for suffixes, pos in suffix_rules:
                if any(decoded.endswith(s) for s in suffixes):
                    pos_map[token_id] = pos
                    matched = True
                    break
            if not matched:
                pos_map[token_id] = "OTHER"

    for token_id in range(len(vocab)):
        if token_id not in pos_map:
            pos_map[token_id] = "OTHER"

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({str(k): v for k, v in pos_map.items()}, f)

    return pos_map


def _compute_embedding_neighbors(model, n_neighbors=50):
    """Compute pairwise cosine similarities from the model's embedding layer.

    Returns a dict mapping each token_id to its top-N neighbor token_ids.
    This is computed lazily and only for requested target tokens.
    """
    embed_weight = model.vocab_embed.embedding.weight.data.float()
    norms = embed_weight.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normalized = embed_weight / norms
    return normalized, n_neighbors


class TokenGroupTracker:
    """Tracks semantic and syntactic probability groupings during denoising."""

    def __init__(self, tokenizer, model, device, n_semantic_neighbors=50):
        self.tokenizer = tokenizer
        self.device = device
        self.vocab_size = len(tokenizer.get_vocab())
        self.n_neighbors = n_semantic_neighbors

        # POS cache
        cache_path = CACHE_DIR / "pos_cache.json"
        if cache_path.exists():
            with open(cache_path) as f:
                raw = json.load(f)
            self.pos_map = {int(k): v for k, v in raw.items()}
        else:
            self.pos_map = _build_pos_cache(tokenizer, cache_path)

        # Build POS group index: category -> list of token_ids
        self.pos_groups = {}
        for tid, cat in self.pos_map.items():
            self.pos_groups.setdefault(cat, []).append(tid)
        for cat in self.pos_groups:
            self.pos_groups[cat] = torch.tensor(self.pos_groups[cat], dtype=torch.long, device=device)

        # Embedding neighbors (lazy computation per target)
        self._embed_normalized, self._n_neighbors = _compute_embedding_neighbors(model, n_semantic_neighbors)
        self._embed_normalized = self._embed_normalized.to(device)
        self._neighbor_cache = {}

    def get_semantic_neighbors(self, target_token_id):
        """Get the top-N semantically similar tokens by embedding cosine similarity."""
        if target_token_id in self._neighbor_cache:
            return self._neighbor_cache[target_token_id]

        target_embed = self._embed_normalized[target_token_id]
        sims = torch.mv(self._embed_normalized, target_embed)
        # Exclude the target itself
        sims[target_token_id] = -1.0
        _, top_ids = sims.topk(self._n_neighbors)
        self._neighbor_cache[target_token_id] = top_ids
        return top_ids

    def compute_group_metrics(self, probs, target_token_id):
        """Compute syntactic and semantic group metrics from a probability distribution.

        Args:
            probs: [V] tensor of normalized probabilities (raw or sampler)
            target_token_id: ground-truth token id

        Returns:
            dict with p_syntactic_group, p_semantic_neighbors, syntactic_rank, semantic_rank
        """
        if target_token_id is None:
            return {
                "p_syntactic_group": None,
                "p_semantic_neighbors": None,
                "syntactic_rank": None,
                "semantic_rank": None,
            }

        target_pos = self.pos_map.get(target_token_id, "OTHER")
        pos_token_ids = self.pos_groups.get(target_pos, torch.tensor([], dtype=torch.long, device=self.device))

        probs_cpu = probs[:self.vocab_size] if probs.shape[0] > self.vocab_size else probs

        # Syntactic group probability and rank
        if len(pos_token_ids) > 0:
            valid_ids = pos_token_ids[pos_token_ids < probs_cpu.shape[0]]
            p_syntactic = probs_cpu[valid_ids].sum().item()
            group_probs = probs_cpu[valid_ids]
            target_p = probs_cpu[target_token_id].item()
            syntactic_rank = int((group_probs > target_p).sum().item()) + 1
        else:
            p_syntactic = 0.0
            syntactic_rank = -1

        # Semantic neighbors probability and rank
        neighbor_ids = self.get_semantic_neighbors(target_token_id)
        valid_neighbor_ids = neighbor_ids[neighbor_ids < probs_cpu.shape[0]]
        p_semantic = probs_cpu[valid_neighbor_ids].sum().item()
        neighbor_probs = probs_cpu[valid_neighbor_ids]
        target_p = probs_cpu[target_token_id].item()
        semantic_rank = int((neighbor_probs > target_p).sum().item()) + 1

        return {
            "p_syntactic_group": p_syntactic,
            "p_semantic_neighbors": p_semantic,
            "syntactic_rank": syntactic_rank,
            "semantic_rank": semantic_rank,
        }
