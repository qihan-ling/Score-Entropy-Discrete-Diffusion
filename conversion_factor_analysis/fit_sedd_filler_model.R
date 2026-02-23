#!/usr/bin/env Rscript
#
# Fit SEDD filler conversion-factor model using the SAME SPR reading-time
# data and methodology as the original SAP benchmark.
#
# Steps:
#   1. Load pre-saved GPT-2 filler model → extract original coefficients
#   2. Load SPR data (reconstructed from model frame)
#   3. Load SEDD word-level surprisals
#   4. Z-score, create lagged predictors, merge with SPR data
#   5. Fit SEDD lme4 model matching the original formula exactly
#   6. Compare conversion factors (ms/bit)

library(lme4)
library(dplyr)
library(stringr)

cat("================================================================\n")
cat("  SEDD vs GPT-2 Conversion Factor Analysis (SPR data)\n")
cat("================================================================\n\n")

# ── Paths ──────────────────────────────────────────────────────────
args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 0) {
  base_dir <- args[1]
} else {
  base_dir <- getwd()
}
sedd_word_csv <- file.path(base_dir, "conversion_factor_analysis", "sedd_filler_word.csv")
spr_csv       <- file.path(base_dir, "conversion_factor_analysis", "Fillers_SPR_reconstructed.csv")
gpt2_model_rds <- file.path(base_dir, "sapbenchmark", "Surprisals", "analysis",
                             "filler_models", "filler_gpt2_sum.rds")
gpt2_scaled    <- file.path(base_dir, "sapbenchmark", "Surprisals", "data",
                             "gpt2", "items_filler.gpt2.csv.scaled")
gpt2_dir <- file.path(base_dir, "sapbenchmark", "Surprisals", "data", "gpt2")

out_dir <- file.path(base_dir, "conversion_factor_analysis")


# ── 1. Extract original GPT-2 coefficients ────────────────────────
cat("[1] Extracting original GPT-2 fixed effects...\n")
gpt2_mod <- readRDS(gpt2_model_rds)
class(gpt2_mod) <- "lmerMod"
gpt2_fe <- fixef(gpt2_mod)

# Compute global SD of GPT-2 surprisal (across all SAP subsets, as rescale.py does)
gpt2_files <- list.files(gpt2_dir, pattern = "[.]csv$", full.names = TRUE)
gpt2_files <- gpt2_files[!grepl("[.]post[.]", gpt2_files)]
gpt2_files <- gpt2_files[!grepl("[.]scaled$", gpt2_files)]
all_gpt2_surps <- c()
for (f in gpt2_files) {
  d <- read.csv(f)
  all_gpt2_surps <- c(all_gpt2_surps, d$sum_surprisal)
}
gpt2_sd_nats <- sd(all_gpt2_surps)
gpt2_sd_bits <- gpt2_sd_nats / log(2)

cat(sprintf("  GPT-2 global SD: %.4f nats, %.4f bits\n", gpt2_sd_nats, gpt2_sd_bits))
cat("  GPT-2 fixed effects (ms/SD):\n")
surp_terms <- c("surprisal_s", "surprisal_p1_s", "surprisal_p2_s", "surprisal_p3_s")
for (term in surp_terms) {
  cat(sprintf("    %s: %.4f ms/SD → %.4f ms/bit\n",
              term, gpt2_fe[term], gpt2_fe[term] / gpt2_sd_bits))
}

# ── 2. Load SPR data ──────────────────────────────────────────────
cat("\n[2] Loading SPR data...\n")
spr <- read.csv(spr_csv)
cat(sprintf("  %d rows, %d participants, %d items\n",
            nrow(spr), length(unique(spr$participant)), length(unique(spr$item))))

# ── 3. Load SEDD word-level surprisals ────────────────────────────
cat("\n[3] Loading SEDD word-level surprisals...\n")
sedd <- read.csv(sedd_word_csv)
sedd$word_pos_1idx <- sedd$word_pos + 1   # Convert to 1-indexed for SPR merge
cat(sprintf("  %d word rows, %d items\n", nrow(sedd), length(unique(sedd$item))))

# ── 4. Z-score SEDD features ─────────────────────────────────────
cat("\n[4] Z-scoring SEDD features...\n")
sedd$sum_surprisal_s <- scale(sedd$sum_surprisal)[,1]
sedd$logfreq_s       <- scale(sedd$logfreq)[,1]
sedd$length_s        <- scale(sedd$length)[,1]

sedd_sd_nats <- sd(sedd$sum_surprisal)
sedd_sd_bits <- sedd_sd_nats / log(2)
cat(sprintf("  SEDD surprisal SD: %.4f nats, %.4f bits\n", sedd_sd_nats, sedd_sd_bits))
cat(sprintf("  SEDD surprisal mean: %.4f nats\n", mean(sedd$sum_surprisal)))

# Rename for merging
sedd_for_merge <- sedd %>%
  select(item, word_pos_1idx, sum_surprisal_s, logfreq_s, length_s) %>%
  rename(surprisal_s = sum_surprisal_s,
         WordPosition = word_pos_1idx)

# ── 5. Merge SPR + SEDD surprisals ───────────────────────────────
cat("\n[5] Merging SPR data with SEDD surprisals...\n")
merged <- merge(x = spr, y = sedd_for_merge,
                by.x = c("item", "WordPosition"),
                by.y = c("item", "WordPosition"),
                all.x = TRUE)

cat(sprintf("  After merge: %d rows\n", nrow(merged)))
cat(sprintf("  Matched rows: %d / %d\n",
            sum(!is.na(merged$surprisal_s)), nrow(merged)))

# ── 6. Create lagged predictors ──────────────────────────────────
cat("\n[6] Creating lagged predictors...\n")
with_lags <- merged %>%
  group_by(item, participant) %>%
  arrange(WordPosition) %>%
  mutate(
    surprisal_p1_s = lag(surprisal_s),
    surprisal_p2_s = lag(surprisal_p1_s),
    surprisal_p3_s = lag(surprisal_p2_s),
    length_p1_s = lag(length_s),
    length_p2_s = lag(length_p1_s),
    length_p3_s = lag(length_p2_s),
    logfreq_p1_s = lag(logfreq_s),
    logfreq_p2_s = lag(logfreq_p1_s),
    logfreq_p3_s = lag(logfreq_p2_s)
  ) %>%
  ungroup()

# Compute sentence length for last-word exclusion
with_lags$sent_length <- sapply(
  str_split(with_lags$Sentence, " "), length
)

# Drop rows with missing predictors (first 3 words) and last word
dropped <- subset(with_lags,
                  !is.na(surprisal_s) &
                    !is.na(surprisal_p1_s) &
                    !is.na(surprisal_p2_s) &
                    !is.na(surprisal_p3_s) &
                    !is.na(logfreq_s) &
                    !is.na(logfreq_p1_s) &
                    !is.na(logfreq_p2_s) &
                    !is.na(logfreq_p3_s) &
                    !is.na(RT) &
                    (sent_length != WordPosition))

cat(sprintf("  After exclusions: %d rows (dropped %d)\n",
            nrow(dropped), nrow(with_lags) - nrow(dropped)))
cat(sprintf("  Unique participants: %d\n", length(unique(dropped$participant))))
cat(sprintf("  Unique items: %d\n", length(unique(dropped$item))))

# ── 7. Fit SEDD model ────────────────────────────────────────────
cat("\n[7] Fitting SEDD lme4 model...\n")
cat("  Formula: RT ~ surprisal_s + surprisal_p1_s + surprisal_p2_s + surprisal_p3_s +\n")
cat("           scale(WordPosition) + logfreq_s*length_s + logfreq_p1_s*length_p1_s +\n")
cat("           logfreq_p2_s*length_p2_s + logfreq_p3_s*length_p3_s +\n")
cat("           (1 + surprisal_s + surprisal_p1_s + surprisal_p2_s + surprisal_p3_s || participant) + (1 | item)\n")
cat("  Optimizer: bobyqa, maxfun=200000\n\n")

sedd_model <- lmer(
  data = dropped,
  RT ~ surprisal_s + surprisal_p1_s + surprisal_p2_s + surprisal_p3_s +
    scale(WordPosition) + logfreq_s*length_s + logfreq_p1_s*length_p1_s +
    logfreq_p2_s*length_p2_s + logfreq_p3_s*length_p3_s +
    (1 + surprisal_s + surprisal_p1_s + surprisal_p2_s + surprisal_p3_s || participant) + (1 | item),
  control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
)

cat("  Model fitted successfully.\n")

# Save model
saveRDS(sedd_model, file.path(out_dir, "filler_sedd_sum.rds"))
cat("  Saved: filler_sedd_sum.rds\n")

# ── 8. Extract and compare conversion factors ────────────────────
cat("\n================================================================\n")
cat("  CONVERSION FACTOR COMPARISON\n")
cat("================================================================\n\n")

sedd_fe <- fixef(sedd_model)

cat(sprintf("%-20s %12s %12s %12s %12s\n",
            "", "GPT-2 ms/SD", "GPT-2 ms/bit", "SEDD ms/SD", "SEDD ms/bit"))
cat(sprintf("%-20s %12s %12s %12s %12s\n",
            "", "-----------", "------------", "----------", "-----------"))
for (term in surp_terms) {
  cat(sprintf("%-20s %12.4f %12.4f %12.4f %12.4f\n",
              term,
              gpt2_fe[term], gpt2_fe[term] / gpt2_sd_bits,
              sedd_fe[term], sedd_fe[term] / sedd_sd_bits))
}

# Significance from model summary
cat("\n=== SEDD Model Summary (surprisal terms) ===\n")
sedd_summary <- summary(sedd_model)
coef_table <- coef(sedd_summary)
for (term in surp_terms) {
  est <- coef_table[term, "Estimate"]
  se  <- coef_table[term, "Std. Error"]
  t   <- coef_table[term, "t value"]
  # Approximate p-value using normal distribution (lme4 doesn't give exact p)
  p   <- 2 * pnorm(-abs(t))
  sig <- ifelse(p < 0.001, "***", ifelse(p < 0.01, "**", ifelse(p < 0.05, "*", "")))
  cat(sprintf("  %s: coef=%.4f, SE=%.4f, t=%.2f, p=%.4e %s → %.4f ms/bit\n",
              term, est, se, t, p, sig, est / sedd_sd_bits))
}

cat("\n=== GPT-2 Model Summary (surprisal terms, from original saved model) ===\n")
gpt2_summary <- summary(gpt2_mod)
gpt2_coef <- coef(gpt2_summary)
for (term in surp_terms) {
  est <- gpt2_coef[term, "Estimate"]
  se  <- gpt2_coef[term, "Std. Error"]
  t   <- gpt2_coef[term, "t value"]
  p   <- 2 * pnorm(-abs(t))
  sig <- ifelse(p < 0.001, "***", ifelse(p < 0.01, "**", ifelse(p < 0.05, "*", "")))
  cat(sprintf("  %s: coef=%.4f, SE=%.4f, t=%.2f, p=%.4e %s → %.4f ms/bit\n",
              term, est, se, t, p, sig, est / gpt2_sd_bits))
}

# Full model summary saved to file
sink(file.path(out_dir, "sedd_spr_model_summary.txt"))
cat("=== SEDD Filler Model (SPR data, lme4) ===\n\n")
print(summary(sedd_model))
sink()

# Convergence warnings
cat("\n=== Convergence check ===\n")
if (length(sedd_model@optinfo$conv$lme4$messages) > 0) {
  cat("  WARNINGS:\n")
  for (msg in sedd_model@optinfo$conv$lme4$messages) cat("    ", msg, "\n")
} else {
  cat("  Model converged without warnings.\n")
}

# ── 9. Final comparison table (ms/bit) ───────────────────────────
cat("\n================================================================\n")
cat("  FINAL COMPARISON TABLE (ms/bit)\n")
cat("================================================================\n\n")

labels <- c("surprisal w_n", "surprisal w_{n-1}", "surprisal w_{n-2}", "surprisal w_{n-3}")
cat(sprintf("%-25s %12s %12s %12s\n", "", "Paper (GPT-2)", "Model (GPT-2)", "SEDD"))
cat(sprintf("%-25s %12s %12s %12s\n", "", "-------------", "-------------", "----"))
paper_vals <- c(1.12, 1.12, 0.58, 0.24)
for (i in 1:4) {
  cat(sprintf("%-25s %12.2f %12.4f %12.4f\n",
              labels[i], paper_vals[i],
              gpt2_fe[surp_terms[i]] / gpt2_sd_bits,
              sedd_fe[surp_terms[i]] / sedd_sd_bits))
}

cat("\n  Note: SEDD surprisals z-scored across filler items only.\n")
cat("  GPT-2 surprisals z-scored across all SAP subsets (original methodology).\n")
cat("  Both models use SAME SPR reading-time data.\n")
cat("  Both models use the same lme4 formula and bobyqa optimizer.\n")
cat("\nDone.\n")
