# Filler conversion model: replicate sapbenchmark Fillers_analysis.R
# but adding SEDD denoising metrics (steps_to_commit, cumulative_kl) as predictors.
#
# Reads:
#   LTR_SAP/analysis/data/sedd_spr_merged.csv (from plan_c_alignment.py)
#   or LTR_SAP/analysis/data/sedd_word_metrics.csv + SPR data
#
# Produces:
#   LTR_SAP/analysis/data/filler_model_coefficients.csv
#   LTR_SAP/analysis/data/filler_model_comparison.csv
#
# Usage:
#   Rscript LTR_SAP/analysis/filler_conversion.R

library(lme4)
library(dplyr)
library(stringr)
library(tidyr)

# --- Load data ---
cat("Loading merged SPR + SEDD data...\n")
spr_merged <- read.csv("LTR_SAP/analysis/data/sedd_spr_merged.csv")

# Filter to fillers only
fillers <- spr_merged[spr_merged$subset == "filler", ]
cat(sprintf("  Filler rows: %d\n", nrow(fillers)))

if (nrow(fillers) == 0) {
    stop("No filler data found. Run plan_c_alignment.py first.")
}

# --- Prepare lags ---
cat("Preparing spillover lags...\n")

prepare_with_lags <- function(df) {
    df <- df %>%
        group_by(item, participant) %>%
        arrange(WordPosition) %>%
        mutate(
            surprisal_p1_s = lag(gpt2_surprisal_s),
            surprisal_p2_s = lag(surprisal_p1_s),
            surprisal_p3_s = lag(surprisal_p2_s),
            length_p1_s = lag(length_s),
            length_p2_s = lag(length_p1_s),
            length_p3_s = lag(length_p2_s),
            logfreq_p1_s = lag(logfreq_s),
            logfreq_p2_s = lag(logfreq_p1_s),
            logfreq_p3_s = lag(logfreq_p2_s),
            steps_p1_s = lag(steps_to_commit_s),
            steps_p2_s = lag(steps_p1_s),
            steps_p3_s = lag(steps_p2_s),
            cumkl_p1_s = lag(cumulative_kl_s),
            cumkl_p2_s = lag(cumkl_p1_s),
            cumkl_p3_s = lag(cumkl_p2_s)
        ) %>%
        ungroup()

    df$sent_length <- sapply(str_split(df$Sentence, " "), length)

    dropped <- subset(df,
        !is.na(gpt2_surprisal_s) &
        !is.na(surprisal_p1_s) & !is.na(surprisal_p2_s) & !is.na(surprisal_p3_s) &
        !is.na(logfreq_s) & !is.na(logfreq_p1_s) &
        !is.na(logfreq_p2_s) & !is.na(logfreq_p3_s) &
        (sent_length != WordPosition)
    )
    cat(sprintf("  After dropping NAs and sentence-final: %d rows\n", nrow(dropped)))
    return(dropped)
}

fillers_prepped <- prepare_with_lags(fillers)

# --- Model 1: GPT-2 surprisal only (replicating sapbenchmark) ---
cat("\nFitting Model 1: GPT-2 surprisal only...\n")
model_gpt2 <- lmer(
    RT ~ gpt2_surprisal_s + surprisal_p1_s + surprisal_p2_s + surprisal_p3_s +
        scale(WordPosition) + logfreq_s * length_s +
        logfreq_p1_s * length_p1_s + logfreq_p2_s * length_p2_s +
        logfreq_p3_s * length_p3_s +
        (1 + gpt2_surprisal_s + surprisal_p1_s + surprisal_p2_s + surprisal_p3_s || participant) +
        (1 | item),
    data = fillers_prepped,
    control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
)
cat("  Model 1 fitted.\n")

# --- Model 2: SEDD steps_to_commit only ---
cat("Fitting Model 2: SEDD steps only...\n")
model_steps <- tryCatch({
    lmer(
        RT ~ steps_to_commit_s + steps_p1_s + steps_p2_s + steps_p3_s +
            scale(WordPosition) + logfreq_s * length_s +
            logfreq_p1_s * length_p1_s + logfreq_p2_s * length_p2_s +
            logfreq_p3_s * length_p3_s +
            (1 + steps_to_commit_s + steps_p1_s + steps_p2_s + steps_p3_s || participant) +
            (1 | item),
        data = fillers_prepped,
        control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
    )
}, error = function(e) {
    cat(sprintf("  Model 2 failed: %s\n", e$message))
    NULL
})

# --- Model 3: GPT-2 surprisal + SEDD steps ---
cat("Fitting Model 3: GPT-2 + SEDD steps...\n")
model_combined <- tryCatch({
    lmer(
        RT ~ gpt2_surprisal_s + surprisal_p1_s + surprisal_p2_s + surprisal_p3_s +
            steps_to_commit_s + steps_p1_s + steps_p2_s + steps_p3_s +
            scale(WordPosition) + logfreq_s * length_s +
            logfreq_p1_s * length_p1_s + logfreq_p2_s * length_p2_s +
            logfreq_p3_s * length_p3_s +
            (1 + gpt2_surprisal_s + steps_to_commit_s || participant) +
            (1 | item),
        data = fillers_prepped,
        control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
    )
}, error = function(e) {
    cat(sprintf("  Model 3 failed: %s\n", e$message))
    NULL
})

# --- Model 4: SEDD cumulative KL only ---
cat("Fitting Model 4: Cumulative KL only...\n")
model_kl <- tryCatch({
    lmer(
        RT ~ cumulative_kl_s + cumkl_p1_s + cumkl_p2_s + cumkl_p3_s +
            scale(WordPosition) + logfreq_s * length_s +
            logfreq_p1_s * length_p1_s + logfreq_p2_s * length_p2_s +
            logfreq_p3_s * length_p3_s +
            (1 + cumulative_kl_s + cumkl_p1_s + cumkl_p2_s + cumkl_p3_s || participant) +
            (1 | item),
        data = fillers_prepped,
        control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
    )
}, error = function(e) {
    cat(sprintf("  Model 4 failed: %s\n", e$message))
    NULL
})

# --- Extract and compare ---
cat("\n--- Model Comparison ---\n")
models <- list(
    gpt2_only = model_gpt2,
    steps_only = model_steps,
    gpt2_plus_steps = model_combined,
    cumkl_only = model_kl
)

comparison <- data.frame(model = character(), AIC = numeric(), BIC = numeric(),
                          logLik = numeric(), stringsAsFactors = FALSE)
coefficients_list <- list()

for (name in names(models)) {
    m <- models[[name]]
    if (is.null(m)) {
        cat(sprintf("  %s: FAILED\n", name))
        next
    }
    aic <- AIC(m)
    bic <- BIC(m)
    ll <- logLik(m)
    cat(sprintf("  %s: AIC=%.1f, BIC=%.1f, logLik=%.1f\n", name, aic, bic, as.numeric(ll)))

    comparison <- rbind(comparison, data.frame(
        model = name, AIC = aic, BIC = bic, logLik = as.numeric(ll)
    ))

    coefs <- summary(m)$coefficients
    coef_df <- data.frame(
        model = name,
        term = rownames(coefs),
        estimate = coefs[, "Estimate"],
        std_error = coefs[, "Std. Error"],
        t_value = coefs[, "t value"]
    )
    coefficients_list[[name]] <- coef_df
}

# Save results
coef_all <- do.call(rbind, coefficients_list)
write.csv(coef_all, "LTR_SAP/analysis/data/filler_model_coefficients.csv", row.names = FALSE)
write.csv(comparison, "LTR_SAP/analysis/data/filler_model_comparison.csv", row.names = FALSE)

cat("\nSaved filler_model_coefficients.csv and filler_model_comparison.csv\n")

# Print conversion factors for the GPT-2 model
cat("\n--- Conversion factors (Model 1: GPT-2 only) ---\n")
gpt2_coef <- summary(model_gpt2)$coefficients["gpt2_surprisal_s", "Estimate"]
cat(sprintf("  GPT-2 surprisal coefficient: %.3f ms/SD\n", gpt2_coef))

if (!is.null(model_steps)) {
    steps_coef <- summary(model_steps)$coefficients["steps_to_commit_s", "Estimate"]
    cat(sprintf("  Steps coefficient: %.3f ms/SD\n", steps_coef))
}

cat("\nDone.\n")
