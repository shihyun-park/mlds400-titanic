# Titanic survival prediction with logistic regression (R).
#
# Expects the Kaggle files in DATA_DIR (default /app/data inside Docker):
#   train.csv, test.csv, and optionally gender_submission.csv
# Writes predictions to DATA_DIR/predictions_r.csv

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
})

DATA_DIR <- Sys.getenv("DATA_DIR", "/app/data")
SEED <- 42
# Different feature set from the Python model:
#  - FamilySize replaces SibSp + Parch (FamilySize = SibSp + Parch + 1, so using
#    all three is perfectly collinear in glm)
#  - Title is used only to impute Age, not as a predictor (Mr/Mrs/Miss/Master
#    nearly duplicate Sex and cause quasi-separation in an unpenalized glm)
#  - IsChild (Age < 13) captures the "women and children first" effect
FEATURES <- c("Pclass", "Sex", "Age", "IsChild", "Fare",
              "FamilySize", "IsAlone", "HasCabin", "Embarked")

step <- function(n, msg) {
  line <- strrep("=", 60)
  cat("\n", line, "\n[Step ", n, "] ", msg, "\n", line, "\n", sep = "")
}

load_csv <- function(name, required = TRUE) {
  path <- file.path(DATA_DIR, name)
  if (!file.exists(path)) {
    if (required) {
      stop(path, " not found. Download the data from Kaggle into src/data/ ",
           "(see README), then mount it with -v.", call. = FALSE)
    }
    return(NULL)
  }
  df <- read_csv(path, show_col_types = FALSE)
  cat(sprintf("Loaded %s: %d rows x %d columns\n", name, nrow(df), ncol(df)))
  df
}

extract_title <- function(name) {
  title <- trimws(sub("\\..*$", "", sub("^[^,]*,", "", name)))
  title <- recode(title, Mlle = "Miss", Ms = "Miss", Mme = "Mrs")
  ifelse(title %in% c("Mr", "Mrs", "Miss", "Master"), title, "Rare")
}

acc <- function(truth, pred) mean(truth == pred)

# Learn imputation values from the TRAINING set only (avoid leakage)
fit_fill_values <- function(train) {
  list(
    age_by_title = train %>% group_by(Title) %>%
      summarise(m = median(Age, na.rm = TRUE), .groups = "drop"),
    age_overall = median(train$Age, na.rm = TRUE),
    fare = median(train$Fare, na.rm = TRUE),
    embarked = names(sort(table(train$Embarked), decreasing = TRUE))[1]
  )
}

transform_df <- function(df, fills, label) {
  cat("\n--- Transforming", label, "---\n")

  n <- sum(is.na(df$Age))
  df <- df %>% left_join(fills$age_by_title, by = "Title") %>%
    mutate(Age = coalesce(Age, m, fills$age_overall)) %>% select(-m)
  cat(sprintf("Age: filled %d missing values with median age by Title\n", n))

  n <- sum(is.na(df$Fare))
  df$Fare[is.na(df$Fare)] <- fills$fare
  cat(sprintf("Fare: filled %d missing values with train median (%.2f)\n", n, fills$fare))

  n <- sum(is.na(df$Embarked))
  df$Embarked[is.na(df$Embarked)] <- fills$embarked
  cat(sprintf("Embarked: filled %d missing values with mode ('%s')\n", n, fills$embarked))

  df <- df %>% mutate(
    HasCabin   = as.integer(!is.na(Cabin)),
    FamilySize = SibSp + Parch + 1,
    IsAlone    = as.integer(FamilySize == 1),
    IsChild    = as.integer(Age < 13),
    Sex        = as.integer(Sex == "female"),
    Pclass     = factor(Pclass, levels = 1:3),
    Embarked   = factor(Embarked, levels = c("C", "Q", "S"))
  )
  cat(sprintf("HasCabin: created from Cabin (%d passengers have a cabin); Cabin dropped\n",
              sum(df$HasCabin)))
  cat(sprintf("FamilySize = SibSp + Parch + 1 (mean %.2f); IsAlone: %d passengers travel alone\n",
              mean(df$FamilySize), sum(df$IsAlone)))
  cat(sprintf("IsChild: %d passengers under 13\n", sum(df$IsChild)))
  cat("Sex: encoded female=1, male=0\n")
  cat("Pclass, Embarked: converted to factors (dummy-coded by glm)\n")
  cat(sprintf("Remaining missing values in features: %d\n",
              sum(is.na(df[, FEATURES]))))
  df
}

main <- function() {
  set.seed(SEED)

  step(1, "Load training data")
  train <- load_csv("train.csv")
  print(head(train))

  step(2, "Explore the training data")
  cat(sprintf("Overall survival rate: %.1f%%\n", 100 * mean(train$Survived)))
  na_counts <- colSums(is.na(train))
  cat("\nMissing values per column:\n"); print(na_counts[na_counts > 0])
  cat("\nSurvival rate by Sex:\n")
  print(train %>% group_by(Sex) %>% summarise(rate = round(mean(Survived), 3)))
  cat("\nSurvival rate by Pclass:\n")
  print(train %>% group_by(Pclass) %>% summarise(rate = round(mean(Survived), 3)))

  step(3, "Feature engineering")
  train$Title <- extract_title(train$Name)
  cat("Title: extracted from Name. Counts:\n"); print(table(train$Title))
  cat("\nSurvival rate by Title:\n")
  print(train %>% group_by(Title) %>% summarise(rate = round(mean(Survived), 3)))
  fills <- fit_fill_values(train)
  train <- transform_df(train, fills, "train.csv")
  cat("\nFinal features:", paste(FEATURES, collapse = ", "), "\n")

  step(4, "Train logistic regression")
  formula <- as.formula(paste("Survived ~", paste(FEATURES, collapse = " + ")))
  idx <- sample(seq_len(nrow(train)), size = floor(0.8 * nrow(train)))
  m_val <- glm(formula, data = train[idx, ], family = binomial)
  val_pred <- as.integer(predict(m_val, train[-idx, ], type = "response") > 0.5)
  cat(sprintf("Hold-out validation accuracy (80/20 split): %.4f\n",
              acc(train$Survived[-idx], val_pred)))

  model <- glm(formula, data = train, family = binomial)
  cat("Refit model on all training rows. Coefficients:\n")
  print(round(coef(model), 3))

  step(5, "Accuracy on the training set")
  train_pred <- as.integer(predict(model, train, type = "response") > 0.5)
  cat(sprintf("Training accuracy: %.4f\n", acc(train$Survived, train_pred)))

  step(6, "Load test data and predict")
  test <- load_csv("test.csv")
  test$Title <- extract_title(test$Name)
  test <- transform_df(test, fills, "test.csv")  # uses TRAIN fill values
  preds <- as.integer(predict(model, test, type = "response") > 0.5)
  out <- data.frame(PassengerId = test$PassengerId, Survived = preds)
  out_path <- file.path(DATA_DIR, "predictions_r.csv")
  write_csv(out, out_path)
  cat(sprintf("\nPredicted survival rate on test set: %.1f%%\n", 100 * mean(preds)))
  cat(sprintf("Saved %d predictions to %s\n", nrow(out), out_path))

  step(7, "Accuracy on the test set")
  if ("Survived" %in% names(test)) {
    cat(sprintf("Test accuracy: %.4f\n", acc(test$Survived, preds)))
  } else {
    ref <- load_csv("gender_submission.csv", required = FALSE)
    if (is.null(ref)) {
      cat("test.csv has no labels and gender_submission.csv not found;",
          "submit predictions_r.csv to Kaggle for the true test score.\n")
    } else {
      merged <- inner_join(out, ref, by = "PassengerId", suffix = c("", "_ref"))
      cat("Kaggle does not publish test labels. Comparing against",
          "gender_submission.csv (Kaggle's 'all women survive' baseline)",
          "as a reference label set:\n")
      cat(sprintf("Test accuracy vs. gender_submission.csv: %.4f\n",
                  acc(merged$Survived_ref, merged$Survived)))
      cat("For the official test score, upload predictions_r.csv to Kaggle.\n")
    }
  }
  cat("\nDone.\n")
}

main()