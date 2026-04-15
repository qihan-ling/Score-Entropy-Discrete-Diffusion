# Strict-LTR SEDD Analysis: Full Results

*Generated from all SAP subsets including fillers*

---

## 1. Data Overview

### Agreement
- Items: 18, Conditions: ['UNAGREE', 'AGREE'], Word-level entries: 530
- **Words with actual denoising metrics: 275/530** (255 words had None — never reached by denoising loop)
- Steps-to-commit (all): mean=70.0, sd=142.1, range=[1, 732]
- Weighted steps (all): mean=47.3, sd=106.9
- Surprisal (tracked only, n=275): mean=11.65, sd=6.67
- Entropy (tracked only, n=275): mean=7.31, sd=3.34
- Cumulative KL (tracked only, n=275): mean=0.0284, sd=0.0348
- Token prediction accuracy: 4.0%

### ClassicGP
- Items: 24, Conditions: ['NPS_UAMB', 'NPZ_UAMB', 'MVRR_UAMB'], Word-level entries: 926
- **Words with actual denoising metrics: 269/926** (657 words had None — never reached by denoising loop)
- Steps-to-commit (all): mean=80.4, sd=265.2, range=[1, 993]
- Weighted steps (all): mean=78.0, sd=256.9
- Surprisal (tracked only, n=269): mean=12.04, sd=4.63
- Entropy (tracked only, n=269): mean=9.40, sd=2.53
- Cumulative KL (tracked only, n=269): mean=0.0455, sd=0.0666
- Token prediction accuracy: 0.3%

### RelativeClause
- Items: 24, Conditions: ['RC_Subj', 'RC_Obj'], Word-level entries: 572
- **Words with actual denoising metrics: 321/572** (251 words had None — never reached by denoising loop)
- Steps-to-commit (all): mean=86.5, sd=177.3, range=[1, 776]
- Weighted steps (all): mean=62.1, sd=142.9
- Surprisal (tracked only, n=321): mean=11.00, sd=5.16
- Entropy (tracked only, n=321): mean=7.52, sd=3.16
- Cumulative KL (tracked only, n=321): mean=0.0318, sd=0.0406
- Token prediction accuracy: 2.6%

### AttachmentAmbiguity
- Items: 24, Conditions: ['AttachMulti', 'AttachHigh', 'AttachLow'], Word-level entries: 1383
- **Words with actual denoising metrics: 273/1383** (1110 words had None — never reached by denoising loop)
- Steps-to-commit (all): mean=54.3, sd=220.2, range=[1, 1001]
- Weighted steps (all): mean=52.6, sd=213.3
- Surprisal (tracked only, n=273): mean=12.12, sd=7.46
- Entropy (tracked only, n=273): mean=9.05, sd=2.87
- Cumulative KL (tracked only, n=273): mean=0.0469, sd=0.0655
- Token prediction accuracy: 1.9%

### filler
- Items: 39, Conditions: N/A, Word-level entries: 687
- **Words with actual denoising metrics: 266/687** (421 words had None — never reached by denoising loop)
- Steps-to-commit (all): mean=58.9, sd=150.4, range=[1, 953]
- Weighted steps (all): mean=41.2, sd=119.6
- Surprisal (tracked only, n=266): mean=9.55, sd=7.04
- Entropy (tracked only, n=266): mean=6.18, sd=3.14
- Cumulative KL (tracked only, n=266): mean=0.0341, sd=0.0509
- Token prediction accuracy: 6.6%

**Total experimental words:** 3411 (1138 with metrics)
**Total filler words:** 687

## 2. Condition Effects at Critical Position

### Agreement (AGREE vs UNAGREE)

**WARNING: 9/36 items at critical position have NO denoising metrics (None) — the loop ran out of steps before reaching them.**

| Metric | AGREE | UNAGREE | diff | Cohen's d | t-stat | p-value |
|--------|-------|-------|------|-----------|--------|---------|
| steps_to_commit | 2.28 (3.04) | 2.89 (3.95) | -0.61 | -0.173 | -0.52 | 0.6068 |
| weighted_steps | 2.27 (3.04) | 2.88 (3.94) | -0.61 | -0.174 | -0.52 | 0.6058 |
| surprisal | 10.85 (3.32) | 19.07 (2.51) | -8.22 | -2.775 | -7.20 | 0.0000*** |
| entropy | 6.50 (0.93) | 6.39 (1.14) | +0.11 | 0.107 | 0.28 | 0.7828 |
| cumulative_kl | 0.00 (0.00) | 0.00 (0.00) | -0.00 | -0.228 | -0.59 | 0.5594 |

### ClassicGP (MVRR_UAMB vs NPS_UAMB)

**WARNING: 72/72 items at critical position have NO denoising metrics (None) — the loop ran out of steps before reaching them.**

| Metric | MVRR_UAMB | NPS_UAMB | diff | Cohen's d | t-stat | p-value |
|--------|-------|-------|------|-----------|--------|---------|
| steps_to_commit | 1.00 (0.00) | 1.00 (0.00) | +0.00 | 0.000 | nan | nan |
| weighted_steps | 1.00 (0.00) | 1.00 (0.00) | +0.00 | 0.000 | nan | nan |

*Additional conditions: ['NPZ_UAMB']*

### RelativeClause (RC_Obj vs RC_Subj)

**WARNING: 4/48 items at critical position have NO denoising metrics (None) — the loop ran out of steps before reaching them.**

| Metric | RC_Obj | RC_Subj | diff | Cohen's d | t-stat | p-value |
|--------|-------|-------|------|-----------|--------|---------|
| steps_to_commit | 4.33 (5.43) | 22.25 (29.30) | -17.92 | -0.850 | -2.95 | 0.0050** |
| weighted_steps | 4.30 (5.38) | 21.82 (28.90) | -17.52 | -0.843 | -2.92 | 0.0054** |
| surprisal | 12.17 (3.95) | 12.43 (2.67) | -0.26 | -0.078 | -0.26 | 0.7985 |
| entropy | 6.44 (1.73) | 8.55 (0.69) | -2.11 | -1.633 | -5.41 | 0.0000*** |
| cumulative_kl | 0.00 (0.01) | 0.01 (0.02) | -0.01 | -0.717 | -2.37 | 0.0222* |

### AttachmentAmbiguity (AttachHigh vs AttachLow)

**WARNING: 72/72 items at critical position have NO denoising metrics (None) — the loop ran out of steps before reaching them.**

| Metric | AttachHigh | AttachLow | diff | Cohen's d | t-stat | p-value |
|--------|-------|-------|------|-----------|--------|---------|
| steps_to_commit | 1.12 (0.34) | 1.12 (0.34) | +0.00 | 0.000 | 0.00 | 1.0000 |
| weighted_steps | 1.12 (0.34) | 1.12 (0.34) | +0.00 | 0.000 | 0.00 | 1.0000 |

*Additional conditions: ['AttachMulti']*

## 3. Effect Direction Profile (ROI 0 to +2)

### Agreement

| ROI | Metric | AGREE mean | UNAGREE mean | diff | p-value |
|-----|--------|------|------|------|---------|
| -1 | steps_to_commit | 8.56 | 8.61 | -0.06 | 0.9926 |
| -1 | weighted_steps | 8.48 | 8.53 | -0.06 | 0.9925 |
| -1 | surprisal | 7.39 | 10.33 | -2.94 | 0.0216* |
| +0 | steps_to_commit | 2.28 | 2.89 | -0.61 | 0.6068 |
| +0 | weighted_steps | 2.27 | 2.88 | -0.61 | 0.6058 |
| +0 | surprisal | 10.85 | 19.07 | -8.22 | 0.0000* |
| +1 | steps_to_commit | 1.22 | 1.06 | +0.17 | 0.2443 |
| +1 | weighted_steps | 1.22 | 1.05 | +0.16 | 0.2451 |
| +1 | surprisal | 12.11 | 12.31 | -0.19 | 0.9322 |
| +2 | steps_to_commit | 1.33 | 1.28 | +0.06 | 0.8782 |
| +2 | weighted_steps | 1.33 | 1.28 | +0.05 | 0.8800 |
| +2 | surprisal | 6.70 | 5.53 | +1.17 | 0.4906 |

### ClassicGP

| ROI | Metric | MVRR_UAMB mean | NPS_UAMB mean | diff | p-value |
|-----|--------|------|------|------|---------|
| -1 | steps_to_commit | 1.00 | 1.00 | +0.00 | nan |
| -1 | weighted_steps | 1.00 | 1.00 | +0.00 | nan |
| +0 | steps_to_commit | 1.00 | 1.00 | +0.00 | nan |
| +0 | weighted_steps | 1.00 | 1.00 | +0.00 | nan |
| +1 | steps_to_commit | 1.00 | 1.00 | +0.00 | nan |
| +1 | weighted_steps | 1.00 | 1.00 | +0.00 | nan |
| +2 | steps_to_commit | 1.00 | 1.00 | +0.00 | nan |
| +2 | weighted_steps | 1.00 | 1.00 | +0.00 | nan |

### RelativeClause

| ROI | Metric | RC_Obj mean | RC_Subj mean | diff | p-value |
|-----|--------|------|------|------|---------|
| -1 | steps_to_commit | 11.62 | 116.88 | -105.25 | 0.0000* |
| -1 | weighted_steps | 11.39 | 113.30 | -101.91 | 0.0000* |
| -1 | surprisal | 12.37 | 5.47 | +6.90 | 0.0000* |
| +0 | steps_to_commit | 4.33 | 22.25 | -17.92 | 0.0050* |
| +0 | weighted_steps | 4.30 | 21.82 | -17.52 | 0.0054* |
| +0 | surprisal | 12.17 | 12.43 | -0.26 | 0.7985 |
| +1 | steps_to_commit | 1.96 | 11.21 | -9.25 | 0.0422* |
| +1 | weighted_steps | 1.95 | 11.06 | -9.11 | 0.0429* |
| +1 | surprisal | 17.33 | 2.54 | +14.79 | 0.0000* |
| +2 | steps_to_commit | 1.29 | 6.29 | -5.00 | 0.0112* |
| +2 | weighted_steps | 1.29 | 6.25 | -4.96 | 0.0110* |
| +2 | surprisal | 3.89 | 12.00 | -8.11 | 0.0001* |

### AttachmentAmbiguity

| ROI | Metric | AttachHigh mean | AttachLow mean | diff | p-value |
|-----|--------|------|------|------|---------|
| -1 | steps_to_commit | 1.00 | 1.00 | +0.00 | nan |
| -1 | weighted_steps | 1.00 | 1.00 | +0.00 | nan |
| +0 | steps_to_commit | 1.12 | 1.12 | +0.00 | 1.0000 |
| +0 | weighted_steps | 1.12 | 1.12 | +0.00 | 1.0000 |
| +1 | steps_to_commit | 1.17 | 1.17 | +0.00 | 1.0000 |
| +1 | weighted_steps | 1.17 | 1.17 | +0.00 | 1.0000 |
| +2 | steps_to_commit | 1.17 | 1.17 | +0.00 | 1.0000 |
| +2 | weighted_steps | 1.17 | 1.17 | +0.00 | 1.0000 |

## 4. Factor Decomposition

Gate (dsigma) is constant with --steps 1024. We examine two variable factors:

- **Score sharpness**: proxied by `final_entropy`
- **Context quality**: proxied by `position` (linear in enforce-prefix LTR)

### Agreement (n=312 tokens)

| Target | Factor | Spearman rho | p-value | Pearson r | p-value |
|--------|--------|-------------|---------|-----------|---------|
| steps_taken | final_entropy | -0.067 | 0.2365 | -0.083 | 0.1449 |
| steps_taken | position | -0.864 | 0.0000 | -0.675 | 0.0000 |
| weighted_steps | final_entropy | 0.076 | 0.1782 | 0.114 | 0.0448 |
| weighted_steps | position | -0.773 | 0.0000 | -0.513 | 0.0000 |

- Token prediction accuracy: 9.0%
- At critical position:
  - steps_taken: AGREE=9.50, UNAGREE=9.50
  - weighted_steps: AGREE=9.41, UNAGREE=9.41
  - final_entropy: AGREE=9.12, UNAGREE=9.12
  - t_commitment: AGREE=0.01, UNAGREE=0.01

### ClassicGP (n=269 tokens)

| Target | Factor | Spearman rho | p-value | Pearson r | p-value |
|--------|--------|-------------|---------|-----------|---------|
| steps_taken | final_entropy | 0.543 | 0.0000 | 0.169 | 0.0055 |
| steps_taken | position | -0.876 | 0.0000 | -0.697 | 0.0000 |
| weighted_steps | final_entropy | 0.537 | 0.0000 | 0.169 | 0.0054 |
| weighted_steps | position | -0.877 | 0.0000 | -0.697 | 0.0000 |

- Token prediction accuracy: 1.1%

### RelativeClause (n=334 tokens)

| Target | Factor | Spearman rho | p-value | Pearson r | p-value |
|--------|--------|-------------|---------|-----------|---------|
| steps_taken | final_entropy | 0.120 | 0.0284 | 0.321 | 0.0000 |
| steps_taken | position | -0.848 | 0.0000 | -0.649 | 0.0000 |
| weighted_steps | final_entropy | 0.210 | 0.0001 | 0.489 | 0.0000 |
| weighted_steps | position | -0.727 | 0.0000 | -0.500 | 0.0000 |

- Token prediction accuracy: 5.4%
- At critical position:
  - steps_taken: RC_Obj=4.77, RC_Subj=23.00
  - weighted_steps: RC_Obj=4.73, RC_Subj=22.57
  - final_entropy: RC_Obj=5.71, RC_Subj=8.47
  - t_commitment: RC_Obj=0.01, RC_Subj=0.02

### AttachmentAmbiguity (n=306 tokens)

| Target | Factor | Spearman rho | p-value | Pearson r | p-value |
|--------|--------|-------------|---------|-----------|---------|
| steps_taken | final_entropy | 0.448 | 0.0000 | 0.463 | 0.0000 |
| steps_taken | position | -0.839 | 0.0000 | -0.673 | 0.0000 |
| weighted_steps | final_entropy | 0.444 | 0.0000 | 0.463 | 0.0000 |
| weighted_steps | position | -0.829 | 0.0000 | -0.673 | 0.0000 |

- Token prediction accuracy: 3.9%

## 5. Correlations with Eye-Tracking Data

ET data loaded: 19994 rows, 111 items

### Agreement

### ClassicGP

*No correlations with |rho| > 0.3 found at item level.*

### RelativeClause

*No correlations with |rho| > 0.3 found at item level.*

### AttachmentAmbiguity

### Summary of notable correlations (|rho| > 0.3)

| Subset | Condition | SEDD Metric | ET Metric | rho | p | n |
|--------|-----------|-------------|-----------|-----|---|---|
| Agreement | AGREE | weighted_steps | regout | 0.354 | 0.1501 | 18 |
| Agreement | UNAGREE | steps_to_commit | gp | -0.302 | 0.2239 | 18 |
| AttachmentAmbiguity | AttachHigh | steps_to_commit | ffd | 0.300 | 0.1539 | 24 |
| AttachmentAmbiguity | AttachHigh | steps_to_commit | gz | 0.482 | 0.0170 | 24 |
| AttachmentAmbiguity | AttachHigh | steps_to_commit | gp | 0.300 | 0.1539 | 24 |
| AttachmentAmbiguity | AttachHigh | steps_to_commit | tt | 0.319 | 0.1293 | 24 |
| AttachmentAmbiguity | AttachHigh | weighted_steps | ffd | 0.300 | 0.1539 | 24 |
| AttachmentAmbiguity | AttachHigh | weighted_steps | gz | 0.482 | 0.0170 | 24 |
| AttachmentAmbiguity | AttachHigh | weighted_steps | gp | 0.300 | 0.1539 | 24 |
| AttachmentAmbiguity | AttachHigh | weighted_steps | tt | 0.319 | 0.1293 | 24 |
| AttachmentAmbiguity | AttachLow | steps_to_commit | ffd | 0.573 | 0.0034 | 24 |
| AttachmentAmbiguity | AttachLow | steps_to_commit | gz | 0.464 | 0.0223 | 24 |
| AttachmentAmbiguity | AttachLow | steps_to_commit | gp | 0.319 | 0.1293 | 24 |
| AttachmentAmbiguity | AttachLow | steps_to_commit | tt | 0.337 | 0.1076 | 24 |
| AttachmentAmbiguity | AttachLow | weighted_steps | ffd | 0.573 | 0.0034 | 24 |
| AttachmentAmbiguity | AttachLow | weighted_steps | gz | 0.464 | 0.0223 | 24 |
| AttachmentAmbiguity | AttachLow | weighted_steps | gp | 0.319 | 0.1293 | 24 |
| AttachmentAmbiguity | AttachLow | weighted_steps | tt | 0.337 | 0.1076 | 24 |
| AttachmentAmbiguity | AttachMulti | steps_to_commit | ffd | 0.410 | 0.0469 | 24 |
| AttachmentAmbiguity | AttachMulti | steps_to_commit | gz | 0.464 | 0.0223 | 24 |
| AttachmentAmbiguity | AttachMulti | steps_to_commit | gp | 0.391 | 0.0586 | 24 |
| AttachmentAmbiguity | AttachMulti | steps_to_commit | tt | 0.410 | 0.0469 | 24 |
| AttachmentAmbiguity | AttachMulti | weighted_steps | ffd | 0.410 | 0.0469 | 24 |
| AttachmentAmbiguity | AttachMulti | weighted_steps | gz | 0.464 | 0.0223 | 24 |
| AttachmentAmbiguity | AttachMulti | weighted_steps | gp | 0.391 | 0.0586 | 24 |
| AttachmentAmbiguity | AttachMulti | weighted_steps | tt | 0.410 | 0.0469 | 24 |

### Mean |rho| by SEDD metric (across all subsets/conditions/ET measures)

| SEDD Metric | Mean |rho| | Mean rho | N comparisons |
|-------------|------------|----------|---------------|
| entropy | nan | nan | 0 |
| steps_to_commit | 0.1947 | 0.1029 | 42 |
| surprisal | nan | nan | 0 |
| weighted_steps | 0.2007 | 0.1074 | 42 |

## 6. Steps-to-Commit vs Surprisal (Plan A)

- **All data** (n=1404):
  - Pearson r = 0.401 (p = 2.41e-55)
  - Spearman rho = 0.389 (p = 5.38e-52)

### Per-subset correlations

| Subset | n | Pearson r | Spearman rho |
|--------|---|-----------|--------------|
| Agreement | 275 | -0.046 | 0.137 |
| ClassicGP | 269 | 0.463 | 0.513 |
| RelativeClause | 321 | 0.359 | 0.299 |
| AttachmentAmbiguity | 273 | 0.649 | 0.570 |
| filler | 266 | 0.370 | 0.426 |

- **Weighted steps vs surprisal**: r=0.351, rho=0.349

- **Cumulative KL vs surprisal**: r=0.284

*Scatter plots saved to /Users/qihan/Documents/Score-Entropy-Discrete-Diffusion/LTR_SAP/analysis/figures/plan_a*

## 7. Trajectory Shape Typology (Plan B)

Collected 1545 token-level trajectory features

### Cluster profiles

| Cluster | n | steps_taken | plateau | slope | entropy | cum_kl |
|---------|---|-------------|---------|-------|---------|--------|
| 0 | 169 | 236.5 | 0.18 | 0.0145 | 3.19 | 0.0219 |
| 1 | 522 | 20.4 | 1.00 | -0.0064 | 5.20 | 0.0118 |
| 2 | 254 | 812.6 | 0.17 | 0.0099 | 9.70 | 0.1361 |
| 3 | 600 | 27.4 | 0.99 | 0.0099 | 9.29 | 0.0129 |

### Cluster distribution by subset

| Subset | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 |
|--------|------|------|------|------|
| Agreement | 16.83% | 42.22% | 8.89% | 32.06% |
| AttachmentAmbiguity | 0.93% | 33.64% | 22.43% | 42.99% |
| ClassicGP | 0.00% | 22.14% | 26.57% | 51.29% |
| RelativeClause | 14.16% | 27.73% | 14.75% | 43.36% |
| filler | 21.74% | 42.47% | 10.70% | 25.08% |

## 8. Entropy Profiles Around Critical Position

### Agreement

- steps_to_commit at ROI=0: AGREE=2.28, UNAGREE=2.89 (diff=-0.61)
- weighted_steps at ROI=0: AGREE=2.27, UNAGREE=2.88 (diff=-0.61)
- entropy at ROI=0: AGREE=6.50, UNAGREE=6.39 (diff=+0.11)

### ClassicGP

- steps_to_commit at ROI=0: MVRR_UAMB=1.00, NPS_UAMB=1.00 (diff=+0.00)
- weighted_steps at ROI=0: MVRR_UAMB=1.00, NPS_UAMB=1.00 (diff=+0.00)

### RelativeClause

- steps_to_commit at ROI=0: RC_Obj=4.33, RC_Subj=22.25 (diff=-17.92)
- weighted_steps at ROI=0: RC_Obj=4.30, RC_Subj=21.82 (diff=-17.52)
- entropy at ROI=0: RC_Obj=6.44, RC_Subj=8.55 (diff=-2.11)

### AttachmentAmbiguity

- steps_to_commit at ROI=0: AttachHigh=1.12, AttachLow=1.12 (diff=+0.00)
- weighted_steps at ROI=0: AttachHigh=1.12, AttachLow=1.12 (diff=+0.00)

## 9. SEDD vs GPT-2 Surprisal

| Subset | SEDD Metric | Spearman rho | Pearson r | n |
|--------|-------------|-------------|-----------|---|
| ClassicGP | steps_to_commit | 0.299 | 0.218 | 911 |
| ClassicGP | weighted_steps | 0.274 | 0.218 | 911 |
| ClassicGP | surprisal | 0.908 | 0.950 | 266 |
| ClassicGP | entropy | 0.458 | 0.420 | 266 |
| RelativeClause | steps_to_commit | 0.325 | 0.326 | 550 |
| RelativeClause | weighted_steps | 0.245 | 0.295 | 550 |
| RelativeClause | surprisal | 0.872 | 0.916 | 310 |
| RelativeClause | entropy | 0.272 | 0.246 | 310 |
| AttachmentAmbiguity | steps_to_commit | 0.356 | 0.473 | 102 |
| AttachmentAmbiguity | weighted_steps | 0.354 | 0.473 | 102 |
| AttachmentAmbiguity | surprisal | 0.952 | 0.991 | 24 |
| AttachmentAmbiguity | entropy | 0.898 | 0.679 | 24 |
| filler | steps_to_commit | 0.304 | 0.274 | 586 |
| filler | weighted_steps | 0.281 | 0.149 | 586 |
| filler | surprisal | 0.968 | 0.961 | 247 |
| filler | entropy | 0.233 | 0.185 | 247 |

### Average correlation with GPT-2 surprisal

| SEDD Metric | Mean Spearman | Mean Pearson |
|-------------|---------------|--------------|
| entropy | 0.465 | 0.382 |
| steps_to_commit | 0.321 | 0.323 |
| surprisal | 0.925 | 0.954 |
| weighted_steps | 0.288 | 0.284 |

## 10. Filler Analysis for Conversion Model

- Filler word-level entries: 687
- Words with denoising metrics: 266/687
- Filler items: 39
- Steps-to-commit: mean=58.9, sd=150.4
- Surprisal (tracked only, n=266): mean=9.55, sd=7.04
- Entropy (tracked only, n=266): mean=6.18, sd=3.14

- **Filler steps vs surprisal**: r=0.370, rho=0.426
- Saved filler_word_metrics.csv for R regression model

- **Tracked filler words** (n=266): surprisal mean=9.55, entropy mean=6.18
- Filler words with None metrics (never reached by loop): 421

## 11. Plan C: SEDD Metrics vs SPR Reading Times

Total SPR-SEDD merged rows: 857407

### Correlation of SEDD metrics with SPR RT

**All positions (steps_to_commit, weighted_steps available for all):**

| Subset | Metric | Pearson r | Spearman rho | n |
|--------|--------|-----------|-------------|---|
| filler | steps_to_commit | 0.004* | 0.083 | 340000 |
| filler | weighted_steps | 0.002 | 0.078 | 340000 |
| ClassicGP | steps_to_commit | 0.004* | 0.001 | 301409 |
| ClassicGP | weighted_steps | 0.004* | 0.009 | 301409 |
| RelativeClause | steps_to_commit | 0.000 | -0.003 | 182392 |
| RelativeClause | weighted_steps | -0.002 | -0.008 | 182392 |
| AttachmentAmbiguity | steps_to_commit | 0.039* | 0.139 | 33606 |
| AttachmentAmbiguity | weighted_steps | 0.039* | 0.129 | 33606 |

**Tracked-only positions (where surprisal/entropy are NOT None):**

| Subset | Metric | Pearson r | Spearman rho | n |
|--------|--------|-----------|-------------|---|
| filler | steps_to_commit | 0.008* | 0.078 | 150000 |
| filler | weighted_steps | 0.002 | 0.063 | 150000 |
| filler | surprisal | 0.019* | 0.101 | 150000 |
| filler | entropy | -0.004 | 0.005 | 150000 |
| filler | cumulative_kl | 0.001 | 0.062 | 150000 |
| ClassicGP | steps_to_commit | 0.013* | 0.019 | 88118 |
| ClassicGP | weighted_steps | 0.013* | 0.019 | 88118 |
| ClassicGP | surprisal | 0.009* | 0.049 | 88118 |
| ClassicGP | entropy | 0.002 | -0.018 | 88118 |
| ClassicGP | cumulative_kl | 0.012* | 0.022 | 88118 |
| RelativeClause | steps_to_commit | 0.002 | -0.031 | 102813 |
| RelativeClause | weighted_steps | -0.001 | -0.037 | 102813 |
| RelativeClause | surprisal | 0.032* | 0.053 | 102813 |
| RelativeClause | entropy | -0.010* | -0.038 | 102813 |
| RelativeClause | cumulative_kl | -0.007* | -0.039 | 102813 |
| AttachmentAmbiguity | steps_to_commit | 0.053* | 0.117 | 7908 |
| AttachmentAmbiguity | weighted_steps | 0.053* | 0.118 | 7908 |
| AttachmentAmbiguity | surprisal | 0.062* | 0.110 | 7908 |
| AttachmentAmbiguity | entropy | 0.040* | 0.070 | 7908 |
| AttachmentAmbiguity | cumulative_kl | 0.054* | 0.117 | 7908 |

## 12. Plan C: SEDD Metrics vs Eye-Tracking Measures

Eye-tracking raw data: 19994 rows, 111 items

ET-SEDD merged rows: 128481

### Correlation of SEDD metrics with ET measures (all positions)

| Measure | steps_to_commit r | weighted_steps r | n |
|---------|-------------------|-----------------|---|
| ffd | -0.053* | -0.048* | 78435 |
| gz | -0.109* | -0.100* | 78435 |
| gp | -0.084* | -0.077* | 78435 |
| tt | -0.080* | -0.072* | 78435 |
| regin | 0.207* | 0.194* | 78435 |
| regout | -0.125* | -0.116* | 78435 |

### Tracked-only positions (surprisal/entropy available)

n = 43857 merged rows

| Measure | surprisal r | entropy r | cumulative_kl r |
|---------|------------|-----------|----------------|
| ffd | 0.041* | 0.019* | -0.069* |
| gz | 0.123* | 0.177* | -0.189* |
| gp | 0.077* | 0.112* | -0.180* |
| tt | 0.112* | 0.158* | -0.148* |
| regin | 0.077* | 0.043* | 0.223* |
| regout | -0.039* | -0.007 | -0.127* |

## 13. Plan C: Spillover Analysis

### SPR spillover: SEDD metric at position i vs RT at i+k

| Predictor | Lag | r | p | n |
|-----------|-----|---|---|---|
| steps_to_commit | i+0 | 0.004* | 0.0002 | 857407 |
| steps_to_commit | i+1 | -0.001 | 0.3840 | 857274 |
| steps_to_commit | i+2 | -0.003* | 0.0176 | 857141 |
| steps_to_commit | i+3 | -0.003* | 0.0071 | 857008 |
| weighted_steps | i+0 | 0.004* | 0.0006 | 857407 |
| weighted_steps | i+1 | -0.001 | 0.4240 | 857274 |
| weighted_steps | i+2 | -0.002 | 0.0531 | 857141 |
| weighted_steps | i+3 | -0.002* | 0.0388 | 857008 |

*Spillover figure saved to figures/plan_c/spr_spillover_profile.png*

## 14. Plan C: Condition Effects — SEDD vs Reading Time

### Do SEDD metrics show the same direction as RT for condition contrasts?

| Subset | Contrast | Metric | Mean_A | Mean_B | diff | d | p |
|--------|----------|--------|--------|--------|------|---|---|
| RelativeClause | RC_Obj - RC_Subj | RT | 439.9 | 431.0 | +8.9 | 0.01 | 0.0105* |
| RelativeClause | RC_Obj - RC_Subj | steps_to_commit | 86.2 | 86.2 | -0.1 | -0.00 | 0.9098 |
| RelativeClause | RC_Obj - RC_Subj | weighted_steps | 61.8 | 61.9 | -0.1 | -0.00 | 0.8929 |

### Interpretation

- **RelativeClause** (RC_Obj vs RC_Subj): RT diff = +8.9ms (+), steps diff = -0.1 (-) → OPPOSITE direction

---

## Key Takeaways

### Condition differentiation
- 5/14 metric-subset combinations show significant condition effects at the critical position (p < 0.05)
  - **Agreement** / surprisal: AGREE vs UNAGREE, diff=-8.22, d=-2.77, p=0.0000
  - **RelativeClause** / steps_to_commit: RC_Obj vs RC_Subj, diff=-17.92, d=-0.85, p=0.0050
  - **RelativeClause** / weighted_steps: RC_Obj vs RC_Subj, diff=-17.52, d=-0.84, p=0.0054
  - **RelativeClause** / entropy: RC_Obj vs RC_Subj, diff=-2.11, d=-1.63, p=0.0000
  - **RelativeClause** / cumulative_kl: RC_Obj vs RC_Subj, diff=-0.01, d=-0.72, p=0.0222

### Steps-to-commit vs surprisal
- Overall Pearson correlation: r = 0.401
- Weighted steps vs surprisal: r = 0.351
- Steps and surprisal are moderately correlated, suggesting shared signal
- The difference (residual) between steps and surprisal may capture additional processing difficulty

### Denoising coverage
- Only **1138/3411 (33%)** experimental word positions were actually reached by the denoising loop
- The remaining 2273 positions have **no metrics at all** (None) — the 1024-step budget was exhausted on earlier positions

### Token prediction accuracy
- Experimental items: 1.9% of committed tokens match the target
- This means `steps_to_commit` reflects denoising effort, not just target token difficulty

## Remaining Issues

1. **Most positions have NO denoising metrics**: The strict-LTR loop exhausts its 1024-step 
   budget on the first ~3 tokens. Later positions are committed in the post-loop final denoiser 
   with `final_surprisal=None, final_entropy=None`. Only `steps_to_commit` (=1) is recorded.

2. **Position confound in strict-LTR**: `steps_taken` is strongly correlated with position 
   (later positions commit faster). `weighted_steps` partially addresses this, but the 
   critical-position experiment (running each position from t=0) provides a cleaner comparison.

3. **Low token prediction accuracy**: Most committed tokens don't match targets. The trajectory 
   and denoising effort measures still reflect processing difficulty, but the relationship to 
   specific target tokens is indirect.

4. **Metric-sampler mismatch**: Logged surprisal/entropy are computed from the **normalized raw score**, 
   but the actual token commitment uses a different distribution (`staggered_score * transp_transition` 
   then `sample_categorical`). These can diverge, so surprisal does not directly describe the decision 
   that produced the committed token.

5. **ClassicGP conditions**: Only UAMB conditions are present in the stimuli CSV. The AMB 
   conditions would be needed for a full ambiguity contrast.

6. **Eye-tracking correlations**: Item-level correlations may be weak due to small n per condition. 
   Participant-level analysis with mixed-effects models (via the R script) would be more powerful.

## Next Steps

1. **Critical-position experiment** (highest priority): Run each target position from full noise 
   (step 0) with correct prefix. This gives every position the full step budget and avoids the 
   coverage/positional confound that makes most strict-LTR metrics meaningless.

2. **Fix the metric-sampler mismatch**: Either log surprisal/entropy from the same `probs` 
   distribution used by `sample_categorical`, or add `sampler_surprisal` alongside the current 
   `raw_score_surprisal`. This makes metrics directly interpretable.

3. **Run soft-context and Monte Carlo experiments** to test whether continuous context 
   representations improve predictive power.

4. **Cross-experiment comparison**: After both strict-LTR and critical-position results are available, 
   run `LTR_SAP_comparison/compare_experiments.py`.
