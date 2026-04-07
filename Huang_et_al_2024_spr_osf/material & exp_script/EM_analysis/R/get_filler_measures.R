library(ggplot2)
library(plyr)
library(dplyr)
library(lme4)
library(stringr)

# -----------------------------------------------------------------------
# 1. Load raw eye-movement measure files 
# -----------------------------------------------------------------------
ffd    <- read.csv("/Users/qihan/Documents/Score-Entropy-Discrete-Diffusion/Huang_et_al_2024_spr_osf/material & exp_script/EM_analysis/R/ffd_424.ixs",    header = T)
gz     <- read.csv("/Users/qihan/Documents/Score-Entropy-Discrete-Diffusion/Huang_et_al_2024_spr_osf/material & exp_script/EM_analysis/R/gz_424.ixs",     header = T)
gp     <- read.csv("/Users/qihan/Documents/Score-Entropy-Discrete-Diffusion/Huang_et_al_2024_spr_osf/material & exp_script/EM_analysis/R/gp_424.ixs",     header = T)
tt     <- read.csv("/Users/qihan/Documents/Score-Entropy-Discrete-Diffusion/Huang_et_al_2024_spr_osf/material & exp_script/EM_analysis/R/tt_424.ixs",     header = T)
regin  <- read.csv("/Users/qihan/Documents/Score-Entropy-Discrete-Diffusion/Huang_et_al_2024_spr_osf/material & exp_script/EM_analysis/R/regin_424.ixs",  header = T)
regout <- read.csv("/Users/qihan/Documents/Score-Entropy-Discrete-Diffusion/Huang_et_al_2024_spr_osf/material & exp_script/EM_analysis/R/regout_424.ixs", header = T)

# -----------------------------------------------------------------------
# 2. Columns 5 onward are already uniquely named per measure file
#    (e.g. ffdR1...ffdR25, gzR1...gzR25, etc.) so no renaming needed.
#    We only rename to match the original script's convention so that
#    left_join can disambiguate the shared ID columns (cols 1-4).
# -----------------------------------------------------------------------
numbofmeasures = 6  # e.g., ffd, gz, gp, tt, regin, regout (five kinds of measures)
Pos_info <- read.csv("/Users/qihan/Documents/Score-Entropy-Discrete-Diffusion/Huang_et_al_2024_spr_osf/material & exp_script/EM_analysis/make_cnt/Position_Info.csv",col.names = c("item","cond","ROI"),header=F)
colnames(ffd)[5:ncol(ffd)] <- paste0("ffd",colnames(ffd)[5:ncol(ffd)])  #change the column names to avoid repeated names for different measures
colnames(gz)[5:ncol(gz)] <- paste0("gz",colnames(gz)[5:ncol(gz)])    #change the column names to avoid repeated names for different measures
colnames(gp)[5:ncol(gp)] <- paste0("gp",colnames(gp)[5:ncol(gp)])    #change the column names to avoid repeated names for different measures
colnames(tt)[5:ncol(tt)] <- paste0("tt",colnames(tt)[5:ncol(tt)])   #change the column names to avoid repeated names for different measures
colnames(regin)[5:ncol(regin)] <- paste0("regin",colnames(regin)[5:ncol(regin)])   #change the column names to avoid repeated names for different measures
colnames(regout)[5:ncol(regout)] <- paste0("regout",colnames(regout)[5:ncol(regout)])   #change the column names to avoid repeated names for different measures

# -----------------------------------------------------------------------
# 3. Merge all measures into one wide dataframe
# -----------------------------------------------------------------------
df <- left_join(ffd, gz)
df <- left_join(df, gp)
df <- left_join(df, tt)
df <- left_join(df, regin)
df <- left_join(df, regout)

# -----------------------------------------------------------------------
# 4. Map numeric subject index to session name
# -----------------------------------------------------------------------
subjnumb_sessionname <- read.delim(
  "/Users/qihan/Documents/Score-Entropy-Discrete-Diffusion/Huang_et_al_2024_spr_osf/material & exp_script/EM_analysis/R/files_processed.lst",
  header = F)
colnames(subjnumb_sessionname) <- "Session_name"
Session_name <- unique(unlist(str_split(subjnumb_sessionname$Session_name, ".da1")))
Session_name <- Session_name[Session_name != ""]
subjnumb_sessionname$Session_name <- Session_name
subjnumb_sessionname$subj <- 1:nrow(subjnumb_sessionname)
df <- left_join(df, subjnumb_sessionname)

# -----------------------------------------------------------------------
# 5. Convert numeric condition codes to labels
# -----------------------------------------------------------------------
df$cond <- as.character(df$cond)
df$cond <- revalue(df$cond, c(
  "1"  = "NPS_AMB",    "2"  = "NPS_UAMB",
  "3"  = "NPZ_AMB",    "4"  = "NPZ_UAMB",
  "5"  = "MVRR_AMB",   "6"  = "MVRR_UAMB",
  "7"  = "RC_Subj",    "8"  = "RC_Obj",
  "9"  = "AttachMulti","10" = "AttachHigh",
  "11" = "AttachLow",  "12" = "AGREE",
  "13" = "UNAGREE",    "14" = "FILLER1",
  "15" = "FILLER2"
))

# -----------------------------------------------------------------------
# 6. Load accuracy / comprehension question data
# -----------------------------------------------------------------------
Accuracydata <- data.frame()
for (i in list.files("/Users/qihan/Documents/Score-Entropy-Discrete-Diffusion/Huang_et_al_2024_spr_osf/material & exp_script/EM_analysis/R/RESULTS_FILE")) {
  Accuracydata <- rbind(
    Accuracydata,
    read.delim(paste0("/Users/qihan/Documents/Score-Entropy-Discrete-Diffusion/Huang_et_al_2024_spr_osf/material & exp_script/EM_analysis/R/RESULTS_FILE/", i))
  )
}

Accuracydata <- Accuracydata[, c(1, 2, 4, 5, 6, 7, 8, 9, 12, 14, 15, 18)]
Accuracydata <- Accuracydata[Accuracydata$condition != "Practice", ]
colnames(Accuracydata) <- c("Session_name", "seq", "correct", "RT_question",
                            "endedby", "whole_sentence_readingtime",
                            "sentence", "question", "List", "item", "cond", "actual_seq")
Accuracydata <- Accuracydata[which(!is.na(Accuracydata$Session_name)), ]

# -----------------------------------------------------------------------
# 7. Apply the same trial-level exclusions as the original script
# -----------------------------------------------------------------------

# Subject L_N066: experiment aborted after trial 98
Accuracydata[which(Accuracydata$Session_name == "L_N066" & Accuracydata$seq %in% c(98:104)), ] <- NA
Accuracydata <- Accuracydata[!is.na(Accuracydata$seq), ]
df[which(df$Session_name == "L_N066" & df$seq %in% c(98:104)), ] <- NA
df <- df[which(!is.na(df$Session_name)), ]

# Colgate subjects: remove last two trials
Accuracydata[which(grepl("_C", Accuracydata$Session_name) & Accuracydata$seq %in% c(95, 96)), "seq"] <- NA
Accuracydata <- Accuracydata[!is.na(Accuracydata$seq), ]
df[which(grepl("_C", df$Session_name) & df$seq %in% c(95, 96)), ] <- NA
df <- df[which(!is.na(df$Session_name)), ]

# Merge accuracy info into df
df <- left_join(df, Accuracydata)

# -----------------------------------------------------------------------
# 8. Participant exclusion: accuracy < 0.80 on fillers
# -----------------------------------------------------------------------
acc_by_subj <- aggregate(
  Accuracydata$correct[Accuracydata$cond %in% c("FILLER1", "FILLER2")],
  by  = list(Accuracydata$Session_name[Accuracydata$cond %in% c("FILLER1", "FILLER2")]),
  FUN = mean
)
below_acc0.8 <- acc_by_subj[acc_by_subj$x < 0.8, 1]
df <- df[!df$Session_name %in% below_acc0.8, ]

# -----------------------------------------------------------------------
# 9. Participant exclusion: blink rate < 0.75
# -----------------------------------------------------------------------
blink_rate  <- c()
blinkmuch   <- c()
blink_files <- list.files("/Users/qihan/Documents/Score-Entropy-Discrete-Diffusion/Huang_et_al_2024_spr_osf/material & exp_script/EM_analysis/EBDoc/all_rej")
blink_files <- blink_files[!blink_files %in% paste0(below_acc0.8, ".rej")]

for (name in blink_files) {
  temp <- read.csv(
    paste0("/Users/qihan/Documents/Score-Entropy-Discrete-Diffusion/Huang_et_al_2024_spr_osf/material & exp_script/EM_analysis/EBDoc/all_rej/", name),
    sep = ""
  )
  for (i in 1:(nrow(temp) - 1)) {
    if ((temp$item[i] == temp$item[i + 1]) & (temp$cond[i] == temp$cond[i + 1])) {
      temp[i, 4] <- NA
    }
  }
  if (sum(temp$good_blink[temp$item <= 72], na.rm = T) / 52 < 0.75) {
    blinkmuch <- c(blinkmuch, substring(name, 1, 6))
  } else {
    blink_rate <- c(blink_rate, sum(temp$good_blink[temp$item <= 72], na.rm = T) / 52)
  }
}
df <- df[!df$Session_name %in% blinkmuch, ]

# -----------------------------------------------------------------------
# 10. Additional item/sentence exclusions (same as original)
# -----------------------------------------------------------------------
df <- df[df$item != 107, ]
df <- df[df$sentence != "Research showing that a tiny European river bug called the water boatman may be the loudest animal on earth.", ]
df <- df[!is.na(df$subj), ]

# -----------------------------------------------------------------------
# 11. Filter to FILLER1 and FILLER2 only
# -----------------------------------------------------------------------
filler_df <- df[df$cond %in% c("FILLER1", "FILLER2"), ]

# -----------------------------------------------------------------------
# 12. Keep ID/metadata columns + all word-region measure columns
#     Word-region columns are those prefixed ffd_R / gz_R / gp_R / tt_R /
#     regin_R / regout_R — i.e. every column with "_R" in its name.
# -----------------------------------------------------------------------
id_cols      <- c("subj", "Session_name", "seq", "item", "cond",
                  "sentence", "question", "correct", "RT_question",
                  "whole_sentence_readingtime", "List", "actual_seq")
measure_cols <- grep("^(ffd|gz|gp|tt|regin|regout)R", colnames(filler_df), value = TRUE)

filler_wide <- filler_df[, c(id_cols, measure_cols)]
all_wide     <- df[, c(id_cols, measure_cols)]
# -----------------------------------------------------------------------
# 13. Inspect and save
# -----------------------------------------------------------------------
cat("Filler trials retained:", nrow(filler_wide), "\n")
cat("Subjects retained:     ", length(unique(filler_wide$Session_name)), "\n")
cat("Word-region columns:   ", length(measure_cols), "\n")

# Preview first few rows / columns
print(head(filler_wide[, 1:min(20, ncol(filler_wide))]))

# Save to CSV
write.csv(filler_wide,
          file      = "/Users/qihan/Documents/Score-Entropy-Discrete-Diffusion/Huang_et_al_2024_spr_osf/material & exp_script/EM_analysis/R/filler_wide.csv",
          row.names = FALSE)

cat("\nSaved to filler_wide.csv\n")

## -- All items --
cat("\n=== All items (filler + non-filler) ===\n")
cat("Total trials:          ", nrow(all_wide), "\n")
cat("Subjects retained:     ", length(unique(all_wide$Session_name)), "\n")
cat("Conditions present:    ", paste(unique(all_wide$cond), collapse = ", "), "\n")
cat("Word-region columns:   ", length(measure_cols), "\n")
print(head(all_wide[, 1:min(20, ncol(all_wide))]))

write.csv(all_wide,
          file      = "/Users/qihan/Documents/Score-Entropy-Discrete-Diffusion/Huang_et_al_2024_spr_osf/material & exp_script/EM_analysis/R/all_wide.csv",
          row.names = FALSE)
cat("Saved to all_wide.csv\n")