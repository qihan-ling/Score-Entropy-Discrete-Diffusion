"""
Shared utilities for SEDD psycholinguistic experiments.

Provides:
- Model loading and forward pass helpers (with/without diagonal masking)
- Stimulus loading from SAP CSV files (ClassicGP, Agreement, etc.)
- Target token identification from disambPosition
- Common probability extraction and metric computation
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import GPT2TokenizerFast
from load_model import load_model
from model import utils as mutils
import os


class SEDDModelWrapper:
    """Wraps SEDD model with convenience methods for experiments."""

    def __init__(self, model_name="louaaron/sedd-medium", device='cuda'):
        self.device = torch.device(
            device if torch.cuda.is_available() else 'cpu')
        self.model, self.graph, self.noise = load_model(
            model_name, self.device)
        self.model.eval()
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.score_fn = mutils.get_score_fn(
            self.model, train=False, sampling=True)

    def tokenize(self, sentence, max_len=1024):
        """Tokenize a sentence and return token IDs on device."""
        tokens = self.tokenizer.encode(
            sentence, return_tensors='pt').to(self.device)
        if tokens.shape[1] > max_len:
            tokens = tokens[:, :max_len]
        return tokens

    def get_sigma(self, t_val):
        """Convert scalar time value to sigma tensor."""
        t = torch.tensor([t_val], device=self.device).unsqueeze(1)
        sigma = self.noise(t)[0]
        return sigma

    def forward_no_diagonal_masking(self, tokens, sigma):
        """
        Forward pass that skips the diagonal masking in transformer.py line 401.

        The standard forward pass zeros out P(input_token) at each position,
        which is correct for diffusion training but wrong for probability
        evaluation on clean text.
        """
        indices = tokens
        x = self.model.vocab_embed(indices)
        c = F.silu(self.model.sigma_map(sigma))
        rotary_cos_sin = self.model.rotary_emb(x)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            for i in range(len(self.model.blocks)):
                x = self.model.blocks[i](x, rotary_cos_sin, c, seqlens=None)
            x = self.model.output_layer(x, c)

        if self.model.scale_by_sigma:
            esigm1_log = torch.where(
                sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1
            ).log().to(x.dtype)[:, None, None]
            x = x - esigm1_log - np.log(x.shape[-1] - 1)

        return x

    def logits_to_probs(self, logits, remove_absorb=True):
        """Convert raw logits to proper probability distribution."""
        probs = F.softmax(logits, dim=-1)
        if remove_absorb and self.graph.absorb:
            probs = probs[..., :-1]
            probs = probs / probs.sum(dim=-1, keepdim=True)
        return probs

    def get_target_prob(self, probs, target_token_id, context_pos):
        """
        Get probability of target token from a specific context position.

        Uses probs[context_pos] to predict target_token_id, matching
        the autoregressive convention where position i-1 predicts token i.
        """
        if target_token_id >= probs.shape[-1]:
            return None
        prob = probs[0, context_pos, target_token_id].item()
        return max(prob, 1e-10)


class StimulusLoader:
    """Load and parse SAP stimuli CSV files.

    Supports all SAP benchmark CSV formats:
    - ClassicGP: disambPosition (1-based word pos), ambiguous column
    - Agreement: disambPosition, Main clause subject number
    - AttachmentAmbiguity: disambPosition, conditions AttachMulti/High/Low
    - RelativeClause: targetPosition (uses different column name)
    """

    # Column name variants for the target word position
    POSITION_COLUMNS = ['disambPosition', 'targetPosition']

    def __init__(self, csv_path, tokenizer):
        self.csv_path = csv_path
        self.tokenizer = tokenizer
        self.df = pd.read_csv(csv_path)
        self.name = os.path.splitext(os.path.basename(csv_path))[0]

        self.pos_column = None
        for col in self.POSITION_COLUMNS:
            if col in self.df.columns:
                self.pos_column = col
                break
        if self.pos_column is None:
            raise ValueError(
                f"CSV must have one of {self.POSITION_COLUMNS}. "
                f"Found columns: {list(self.df.columns)}")

    def get_stimuli(self):
        """
        Parse stimuli and identify target tokens.

        Returns list of dicts with fields:
            item, condition, base_condition, sentence,
            disamb_position (1-based word pos), ambiguous (0/1),
            target_word, target_token_pos (0-based), target_token_id,
            target_token_span (list of token positions for multi-token words),
            all_token_ids
        """
        stimuli = []
        skipped = 0
        for _, row in self.df.iterrows():
            sentence = row['Sentence']
            if not isinstance(sentence, str) or sentence.strip() == '':
                skipped += 1
                continue

            disamb_pos = int(row[self.pos_column])
            item = row['item']
            condition = row['condition']

            token_ids = self.tokenizer.encode(sentence)
            words = sentence.split()

            target_info = self._find_target_token_position(
                sentence, words, disamb_pos, token_ids)
            if target_info is None:
                skipped += 1
                continue

            # Parse base condition (NPS/NPZ/MVRR, AttachMulti, RC_Subj, etc.)
            base_cond = condition.split('_')[0] if '_' in condition else condition

            stimulus = {
                'item': item,
                'condition': condition,
                'base_condition': base_cond,
                'sentence': sentence,
                'disamb_position': disamb_pos,
                'target_word': target_info['word'],
                'target_token_pos': target_info['token_pos'],
                'target_token_id': target_info['token_id'],
                'target_token_span': target_info['token_span'],
                'all_token_ids': token_ids,
            }

            # Ambiguity: explicit column or infer from condition name
            if 'ambiguous' in self.df.columns:
                stimulus['ambiguous'] = int(row['ambiguous'])
            else:
                stimulus['ambiguous'] = None

            if 'Main clause subject number' in self.df.columns:
                stimulus['subject_number'] = row['Main clause subject number']

            stimuli.append(stimulus)

        if skipped > 0:
            print(f"  Warning: skipped {skipped} rows (empty or bad mapping)")
        return stimuli

    def _find_target_token_position(self, sentence, words, disamb_pos_1based,
                                     token_ids):
        """
        Map 1-based word position to 0-based token position.

        Uses GPT2TokenizerFast offset mapping for robust alignment between
        whitespace-split words and BPE tokens. Falls back to cumulative
        counting if offset mapping is unavailable.
        """
        if disamb_pos_1based < 1 or disamb_pos_1based > len(words):
            return None

        target_word = words[disamb_pos_1based - 1]

        # Strategy: use offset_mapping from the fast tokenizer
        try:
            encoding = self.tokenizer(
                sentence, return_offsets_mapping=True,
                add_special_tokens=False)
            offsets = encoding['offset_mapping']

            # Find the character span of the target word in the sentence
            char_pos = 0
            for i in range(disamb_pos_1based - 1):
                char_pos = sentence.index(words[i], char_pos) + len(words[i])
            target_char_start = sentence.index(
                target_word, char_pos)

            # Find which token contains target_char_start
            token_pos = None
            for tidx, (start, end) in enumerate(offsets):
                if start <= target_char_start < end:
                    token_pos = tidx
                    break

            if token_pos is None:
                return self._find_target_fallback(
                    words, disamb_pos_1based, token_ids)

            # Find span: all tokens that overlap with the target word
            target_char_end = target_char_start + len(target_word)
            span = [tidx for tidx, (s, e) in enumerate(offsets)
                    if s < target_char_end and e > target_char_start]

            return {
                'word': target_word,
                'token_pos': token_pos,
                'token_id': token_ids[token_pos],
                'token_span': span,
            }
        except Exception:
            return self._find_target_fallback(
                words, disamb_pos_1based, token_ids)

    def _find_target_fallback(self, words, disamb_pos_1based, token_ids):
        """Fallback: cumulative token counting."""
        target_word = words[disamb_pos_1based - 1]
        cumulative_tokens = 0

        for word_idx, word in enumerate(words):
            prefix = " " if word_idx > 0 else ""
            word_tokens = self.tokenizer.encode(prefix + word)
            word_token_count = len(word_tokens)

            if word_idx == disamb_pos_1based - 1:
                token_pos = cumulative_tokens
                if token_pos >= len(token_ids):
                    return None
                return {
                    'word': target_word,
                    'token_pos': token_pos,
                    'token_id': token_ids[token_pos],
                    'token_span': list(range(
                        token_pos, token_pos + word_token_count)),
                }
            cumulative_tokens += word_token_count

        return None


def compute_surprisal(prob):
    """Compute surprisal from probability, with floor for numerical safety."""
    prob = max(prob, 1e-10)
    return -np.log(prob)


def create_output_dir(base_dir="outputs"):
    """Create output directory if it doesn't exist."""
    os.makedirs(base_dir, exist_ok=True)
    return base_dir
