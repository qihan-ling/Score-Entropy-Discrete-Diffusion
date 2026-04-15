# Soft-Context Sequential Rescheduling: Design Document

## 1. The Metric-Sampler Mismatch

### Problem statement

In the strict-LTR experiment, we observe a puzzling pattern: logged `final_surprisal`
is near zero (implying the model is "confident" in the target token), yet only ~2% of
committed tokens actually match the target. This is not a bug -- it reflects a
fundamental mismatch between what the metrics measure and what the sampler uses.

### Two distribution pipelines

The SEDD codebase computes two distinct probability distributions at each denoising
step. They diverge after the raw score leaves the neural network.

```
                         score_fn(x, sigma)
                               |
                     model output (log-scores)
                               |
                           .exp()
                               |
                      raw_score [B, L, V]          <-- "P_raw"
                         /            \
                        /              \
            [Metric Pipeline]     [Sampler Pipeline]
                   |                      |
             normalize(P_raw)      staggered_score(P_raw, dsigma)
                   |                      |
              P_metric [V]         stag_score [B, L, V]
                   |                      |
        surprisal = -log2(             * transp_transition(x, sigma)
          P_metric[target])               |
                   |               probs [B, L, V]
        entropy = H(P_metric)             |
                                   (truncate MASK dim if absorb)
                                          |
                                   sample_categorical(probs)
                                          |
                                   committed_token (int)
```

### Where each distribution comes from

**P_metric (used for logging):** Defined in `sedd_helpers.py`, `compute_frontier_metrics`:

```python
probs = score[0, frontier_pos]      # raw score at frontier
p = probs / probs.sum()             # simple normalization
surprisal = -log2(p[target_token])
entropy = -(p * log2(p)).sum()
```

This is a straightforward normalization of the neural network's exponential output.

**P_sampler (used for commitment):** Defined in `sampling.py`, `Denoiser.update_fn`:

```python
score = score_fn(x, sigma)                                  # same raw score
stag_score = self.graph.staggered_score(score, sigma)       # graph transformation 1
probs = stag_score * self.graph.transp_transition(x, sigma) # graph transformation 2
return sample_categorical(probs)                            # stochastic commitment
```

The two graph transformations reshape the distribution substantially.

### What staggered_score and transp_transition do

For the **Absorbing** graph (used by SEDD):

**`staggered_score(score, dsigma)`** (`graph_lib.py` line 234):
```python
score = score.clone()
extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
score *= dsigma.exp()[:, None]
score[..., -1] += extra_const    # redistributes mass to MASK dimension
```
This multiplies all real-token scores by `e^dsigma` and adds a compensating term to
the MASK dimension. At high sigma (early in denoising), this concentrates mass on MASK,
suppressing real tokens. At low sigma (late in denoising), it approximately preserves
the raw distribution.

**`transp_transition(i, sigma)`** (`graph_lib.py` line 218):
```python
edge = (-sigma).exp() * F.one_hot(i, num_classes=self.dim)
edge += where(i == self.dim - 1, 1 - (-sigma).exp(), 0)[..., None]
```
This produces a matrix where:
- For positions currently holding a real token: mass `e^{-sigma}` on the current token,
  zero elsewhere (the token "stays")
- For MASK positions: mass `1 - e^{-sigma}` spread across all tokens

The element-wise product `stag_score * transp_transition` then means:
- At MASK positions: the sampler distribution is roughly `stag_score * (1 - e^{-sigma})`
  for each real token, giving a reduced but nonzero chance of committing
- At real-token positions: the sampler strongly prefers keeping the current token

### Why surprisal ~ 0 coexists with ~95% wrong commits

1. **P_metric may peak on the target:** The raw normalized score `P_metric` reflects the
   neural network's belief about which token belongs at this position. It often peaks on
   the correct token, giving low surprisal.

2. **P_sampler is distorted by the graph:** After `staggered_score * transp_transition`,
   the distribution is reshaped. At high sigma (early steps), P_sampler has most mass on
   MASK (the absorbing state), with only a thin tail on real tokens. The token that wins
   `sample_categorical` from this thin tail need not match the P_metric argmax.

3. **Stochasticity of `sample_categorical`:** Even when P_sampler has a mode on the
   target, the Gumbel-max trick (exponential sampling) introduces randomness. With many
   vocabulary entries, the probability of sampling exactly the target can be low even if
   it has the highest individual probability.

4. **The post-loop denoiser:** Many positions are never reached by the main denoising
   loop (the 1024-step budget exhausts on early positions). These positions get committed
   in a single final `Denoiser.update_fn` call at the end, with no frontier metrics
   logged (`final_surprisal = None`, `final_entropy = None`). In the analysis pipeline,
   these `None` values were previously converted to 0, creating the illusion of zero
   surprisal.

### Implication for this experiment

This experiment fixes the mismatch by logging metrics from **both** distributions at
every step:
- `entropy`, `p_target` from P_metric (raw score, for comparability with prior work)
- `sampler_entropy`, `sampler_p_target` from P_sampler (the actual decision distribution)

This dual logging makes it possible to study the divergence between "what the network
thinks" and "what the diffusion process does."


## 2. Unified Framework: Sequential Rescheduling with Soft Context

### Motivation

The strict-LTR experiment revealed three critical issues:
1. **Step budget starvation:** 1024 steps are consumed by the first 2-3 tokens; 67% of
   positions get no meaningful metrics.
2. **Positional confound:** Later positions always commit in 1 step, regardless of
   linguistic difficulty.
3. **Metric-sampler mismatch:** Logged metrics don't reflect the actual commitment
   distribution.

The critical-position experiment addresses (1) and (2) by giving each position its own
full 1024-step schedule. But it runs each position in complete isolation with a hard
ground-truth prefix, which doesn't capture how uncertainty at one position propagates to
the next -- a key aspect of incremental human language processing.

This experiment combines the strengths of both approaches into a unified framework.

### Design

**Sequential rescheduling with soft context:** Process the 6 critical positions
(crit-2 through crit+3) one at a time, each with a full 1024-step denoising schedule
starting from t=0. After each position commits, its representation in the prefix for
subsequent positions is a *soft embedding* -- a probability-weighted average of token
embeddings peaked at the ground truth.

**Input construction:**
```
Round 1 (position crit-2):
  [<eot>, tok_1, ..., tok_{crit-3}, MASK, MASK, ..., MASK_pad]
  |------- hard prefix ----------|  ^frontier    |-- padding --|

Round 2 (position crit-1):
  [<eot>, tok_1, ..., tok_{crit-3}, SOFT(crit-2), MASK, ..., MASK_pad]
  |------- hard prefix ---------|  ^soft context   ^frontier

Round 3 (position crit):
  [<eot>, tok_1, ..., tok_{crit-3}, SOFT(crit-2), SOFT(crit-1), MASK, ..., MASK_pad]
  |------- hard prefix ---------|  ^--- soft context ---------^  ^frontier

... and so on for crit+1, crit+2, crit+3
```

Padding extends to a fixed total length (e.g., 256 tokens) so the model cannot infer
sentence length.

### The lambda parameter

After denoising position k, the model's sampler distribution at k is `p_model(v)`. The
soft context distribution is:

```
p_soft(v) = (1 - lambda) * p_model(v) + lambda * delta(v, ground_truth_k)
```

The soft embedding fed to the model is:

```
e_soft_k = sum_v  p_soft(v) * EmbeddingTable[v]
```

**Lambda controls the full spectrum:**

| lambda | Behavior | Equivalent to |
|--------|----------|---------------|
| 1.0 | Hard ground-truth embedding | Critical-position experiment |
| 0.75 | Mostly ground truth, some model uncertainty | -- |
| 0.5 | Equal mixture | -- |
| 0.25 | Mostly model, ground truth anchoring | -- |
| 0.0 | Pure model distribution | Pure uncertainty propagation |

**Ablation values:** {0.0, 0.25, 0.5, 0.75, 1.0}

**Phased rollout:**
1. First run lambda=1.0 and verify results match critical-position experiment (sanity
   check that the wrapper is correct)
2. Then run lambda=0.5 and lambda=0.0 on Agreement subset as a pilot
3. If results are meaningful, run full ablation on all subsets

### Per-step metric tracking

At every denoising step j (0 through 1023), for the active frontier position, log:

| Metric | Source | Description |
|--------|--------|-------------|
| `entropy` | P_metric | Shannon entropy of raw score distribution |
| `p_target` | P_metric | P(ground_truth) under raw score |
| `kl_from_prev` | P_metric | D_KL(P_current \|\| P_previous) |
| `top5_ids` | P_metric | Token IDs of 5 highest-probability tokens |
| `top5_probs` | P_metric | Corresponding probabilities |
| `sampler_entropy` | P_sampler | Shannon entropy of sampler distribution |
| `sampler_p_target` | P_sampler | P(ground_truth) under sampler distribution |

At the commitment step additionally:
- `top50_ids`, `top50_probs` from P_sampler
- `p_model_top50_ids`, `p_model_top50_probs` from P_metric (used to build soft context)

**Storage:** ~15 values x 1024 steps x 6 positions = ~90K values per item (~360KB JSON).

### New analyses enabled by trajectory data

- **Entropy curve shape:** Classify each position's denoising trajectory as cliff
  (sudden drop), gradual descent, plateau-then-drop, or oscillating. These shapes may
  correlate differently with different reading-time measures (FFD vs TT vs GP).
- **Belief convergence point:** The step at which P(gt) first exceeds 0.5, vs. the step
  at which the model actually commits. The gap is "hesitation."
- **KL spike analysis:** Steps with large KL represent "aha moments" where the model
  drastically revises beliefs. Timing relative to the critical position may predict
  garden-path effects.
- **Sampler-raw divergence profile:** How much do P_metric and P_sampler disagree over
  the course of denoising? Does the divergence predict low accuracy?

### Filler handling: sliding window

Filler items have no designated critical position. Use a sliding-window approach:

- For a filler with N words (0-indexed), the 6-token window spans [center-2, center+3].
- Earliest valid center: **word 3** (so crit-2 = word 1 has at least `<eot>` + word 0
  as hard prefix).
- Latest valid center: **word N-4** (so crit+3 = word N-1 is within bounds).
- This yields N-5 windows per filler item.
- Each window is one full run (6 positions x 1024 steps).
- Output: `results/lambda_{value}/filler/item_{id}_window_{center}.json`

For 39 filler items averaging ~17 words: ~470 windows per lambda value.

### Comparison with existing experiments

| Aspect | Strict-LTR | Critical-Position | This experiment |
|--------|-----------|------------------|-----------------|
| Steps per position | ~500/1/1/... | 1024 each | 1024 each |
| Prefix type | All MASK | Hard ground-truth | Soft (lambda) |
| Inter-position dynamics | Yes (all) | No (isolated) | Yes (6-token window) |
| Runs per item | 1 | 6 | 6 (sequential) |
| Total steps per item | 1024 | 6144 | 6144 |
| Sentence length hidden | No | No | Yes (padded) |
| Metric-sampler mismatch | Yes (only P_metric) | Yes | Fixed (both logged) |
| lambda=1.0 equivalent | -- | Yes | Critical-position |


## 3. Implementation Challenges and Solutions

### Challenge 1: The scatter at line 401 of SEDD.forward

**Problem:** `SEDD.forward()` ends with:
```python
x = torch.scatter(x, -1, indices[..., None], torch.zeros_like(x[..., :1]))
```
This zeros the output score at the vocabulary index of the current discrete token for
each position. It implements a structural constraint of the score parameterization
(the score for the current state is set to zero). It requires `indices` to be a tensor
of integer token IDs.

With soft context, there is no single "current token" at denoised positions -- we have a
distribution.

**Solution:** Keep the `indices` tensor discrete at all positions. For soft-context
positions, put the **ground-truth token ID** in `indices`. The scatter zeros the
ground-truth token's score at those positions (same behavior as hard enforce-prefix).
The soft embedding only affects the hidden representations that flow through the
transformer blocks -- it does not change the post-scatter score masking.

This works because:
1. We only read scores at the **frontier** (MASK) position, not at prefix positions
2. The scatter at prefix positions affects scores we never use
3. The soft embedding influences the frontier's score via attention in the transformer
   blocks, which happens *before* the scatter

### Challenge 2: Graph operations assume discrete state

**Problem:** `transp_transition(x, sigma)` and related graph functions use `F.one_hot(i)`
and comparisons like `i == self.dim - 1`, all requiring integer `x`.

**Solution:** No change needed. The state tensor `x` passed to the predictor remains
discrete at all times:
- Prefix positions: ground-truth token IDs (discrete)
- Soft-context positions: ground-truth token IDs (discrete -- the soft embedding lives
  only inside the model's hidden states, not in `x`)
- Frontier position: MASK token ID (until committed)
- Future positions: MASK token ID

The soft embedding injection happens *inside* the model's forward pass (between
`vocab_embed` and the transformer blocks), not in the external state `x`.

### Challenge 3: Which distribution for p_model?

**Problem:** After denoising position k, what distribution should become the soft
context? Options:
1. Raw score distribution (P_metric) -- what the network directly outputs
2. Sampler distribution (P_sampler) -- what actually drove the commitment
3. Distribution at the specific commitment step

**Solution:** Use the **sampler distribution** (P_sampler) at the commitment step. This
is the distribution that:
- Reflects the full diffusion process (including graph transformations)
- Was responsible for the token selection
- Is most self-consistent with the model's internal state at commitment time

```python
# At the step where position k commits:
stag_score = graph.staggered_score(score, sigma)
probs = stag_score * graph.transp_transition(x, sigma)
if graph.absorb:
    probs = probs[..., :-1]  # truncate MASK dim
p_model = probs[0, k] / probs[0, k].sum()  # normalize

p_soft = (1 - lam) * p_model + lam * one_hot(gt_token, vocab_size)
e_soft = p_soft @ model.vocab_embed.embedding[:vocab_size]
```

### Challenge 4: Wrapping the model forward pass

**Problem:** `score_fn` (from `get_score_fn`) calls `model(x, sigma)` directly. We need
to intercept the embedding step to inject soft embeddings without modifying the base
model code.

**Solution:** Create a `SoftContextWrapper` class:

```python
class SoftContextWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._soft_positions = []      # list of int positions
        self._soft_embeddings = None   # [n_soft, hidden_dim] tensor

    def set_soft_context(self, positions, embeddings):
        self._soft_positions = positions
        self._soft_embeddings = embeddings

    def clear_soft_context(self):
        self._soft_positions = []
        self._soft_embeddings = None

    def forward(self, indices, sigma):
        x = self.model.vocab_embed(indices)

        # Inject soft embeddings at specified positions
        if self._soft_positions and self._soft_embeddings is not None:
            for i, pos in enumerate(self._soft_positions):
                x[:, pos] = self._soft_embeddings[i]

        c = F.silu(self.model.sigma_map(sigma))
        rotary_cos_sin = self.model.rotary_emb(x)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for block in self.model.blocks:
                x = block(x, rotary_cos_sin, c, seqlens=None)
            x = self.model.output_layer(x, c)

        if self.model.scale_by_sigma:
            esigm1_log = torch.where(
                sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1
            ).log().to(x.dtype)[:, None, None]
            x = x - esigm1_log - np.log(x.shape[-1] - 1)

        # scatter uses original discrete indices (not soft embeddings)
        x = torch.scatter(x, -1, indices[..., None], torch.zeros_like(x[..., :1]))
        return x
```

This wrapper is passed to `get_score_fn` in place of the raw model.

### Challenge 5: Out-of-distribution input validation

**Problem:** The model was trained exclusively on discrete token embeddings. Soft
embeddings (weighted averages of multiple token embeddings) are out-of-distribution
inputs. The model may produce pathological outputs.

**Solution:** Phased validation approach:

1. **Lambda=1.0 sanity check:** With lambda=1.0, `p_soft = delta(gt)` and the soft
   embedding equals `embed(gt)` exactly. Results should be identical to the
   critical-position experiment. Any discrepancy indicates a wrapper bug.

2. **Embedding space analysis:** Compute the L2 distance between `e_soft` and
   `embed(gt)` for different lambda values. For lambda >= 0.5, the soft embedding
   should be close to the ground-truth embedding in L2 terms, since the mixture is
   dominated by the ground-truth component.

3. **Output stability check:** For a few test items, compare the model's score
   distribution at the frontier position across lambda values. If scores remain
   well-formed (finite, non-negative, roughly normalizable), the model is handling
   soft inputs gracefully.

4. **Gradient-free operation:** Since we never backpropagate through the soft embeddings
   (this is inference only), there is no risk of gradient instability. The only risk is
   the forward pass producing garbage, which is detectable.


## File Structure

```
LTR_SAP_critical_soft/
  DESIGN.md                          # this document
  soft_context_wrapper.py            # SoftContextWrapper class
  transformer_critical_region.py     # core experiment script
  batch_runner_critical_region.py    # batch runner with --lambda argument
  results/
    lambda_1.0/
      Agreement/AGREE/item_1.json
      Agreement/UNAGREE/item_1.json
      ...
      filler/item_73_window_3.json
      filler/item_73_window_4.json
      ...
    lambda_0.5/
      ...
    lambda_0.0/
      ...
```
