"""
Titanic survival prediction with logistic regression (Python).

Expects the Kaggle files in DATA_DIR (default /app/data inside Docker):
    train.csv, test.csv, and optionally gender_submission.csv
Writes predictions to DATA_DIR/predictions_python.csv
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
SEED = 42
FEATURES = [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
    "FamilySize", "IsAlone", "HasCabin",
    "Embarked_C", "Embarked_Q", "Embarked_S",
    "Title_Mr", "Title_Mrs", "Title_Miss", "Title_Master", "Title_Rare",
]


def step(n, msg):
    print(f"\n{'=' * 60}\n[Step {n}] {msg}\n{'=' * 60}")


def load_csv(name, required=True):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        if required:
            sys.exit(
                f"ERROR: {path} not found.\n"
                "Download the data from Kaggle and place it in src/data/ "
                "(see README), then mount it with -v."
            )
        return None
    df = pd.read_csv(path)
    print(f"Loaded {name}: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def extract_title(name):
    title = name.split(",")[1].split(".")[0].strip()
    mapping = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
    title = mapping.get(title, title)
    return title if title in {"Mr", "Mrs", "Miss", "Master"} else "Rare"


def fit_fill_values(train):
    """Learn imputation values from the TRAINING set only (avoid leakage)."""
    return {
        "age_by_title": train.groupby("Title")["Age"].median().to_dict(),
        "age_overall": train["Age"].median(),
        "fare": train["Fare"].median(),
        "embarked": train["Embarked"].mode()[0],
    }


def add_title(df):
    df = df.copy()
    df["Title"] = df["Name"].apply(extract_title)
    return df


def transform(df, fills, label):
    df = df.copy()
    print(f"\n--- Transforming {label} ---")

    n = df["Age"].isna().sum()
    df["Age"] = df.apply(
        lambda r: fills["age_by_title"].get(r["Title"], fills["age_overall"])
        if pd.isna(r["Age"]) else r["Age"], axis=1)
    print(f"Age: filled {n} missing values with median age by Title")

    n = df["Fare"].isna().sum()
    df["Fare"] = df["Fare"].fillna(fills["fare"])
    print(f"Fare: filled {n} missing values with train median ({fills['fare']:.2f})")

    n = df["Embarked"].isna().sum()
    df["Embarked"] = df["Embarked"].fillna(fills["embarked"])
    print(f"Embarked: filled {n} missing values with mode ('{fills['embarked']}')")

    df["HasCabin"] = df["Cabin"].notna().astype(int)
    print(f"HasCabin: created from Cabin ({df['HasCabin'].sum()} passengers have a cabin); Cabin dropped")

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    print(f"FamilySize = SibSp + Parch + 1 (mean {df['FamilySize'].mean():.2f}); "
          f"IsAlone: {df['IsAlone'].sum()} passengers travel alone")

    df["Sex"] = (df["Sex"] == "female").astype(int)
    print("Sex: encoded female=1, male=0")

    for col, levels in [("Embarked", ["C", "Q", "S"]),
                        ("Title", ["Mr", "Mrs", "Miss", "Master", "Rare"])]:
        for lv in levels:
            df[f"{col}_{lv}"] = (df[col] == lv).astype(int)
    print("Embarked, Title: one-hot encoded")

    print(f"Remaining missing values in features: {df[FEATURES].isna().sum().sum()}")
    return df


def main():
    step(1, "Load training data")
    train = load_csv("train.csv")
    print(train.head())

    step(2, "Explore the training data")
    print(f"Overall survival rate: {train['Survived'].mean():.1%}")
    print("\nMissing values per column:")
    print(train.isna().sum()[train.isna().sum() > 0].to_string())
    print("\nSurvival rate by Sex:")
    print(train.groupby("Sex")["Survived"].mean().round(3).to_string())
    print("\nSurvival rate by Pclass:")
    print(train.groupby("Pclass")["Survived"].mean().round(3).to_string())

    step(3, "Feature engineering")
    train = add_title(train)
    print("Title: extracted from Name. Counts:")
    print(train["Title"].value_counts().to_string())
    print("\nSurvival rate by Title:")
    print(train.groupby("Title")["Survived"].mean().round(3).to_string())
    fills = fit_fill_values(train)
    train = transform(train, fills, "train.csv")
    print(f"\nFinal feature set ({len(FEATURES)}): {FEATURES}")

    step(4, "Train logistic regression")
    X, y = train[FEATURES].values, train["Survived"].values
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    model = LogisticRegression(max_iter=1000, random_state=SEED)
    model.fit(scaler.transform(X_tr), y_tr)
    val_acc = accuracy_score(y_val, model.predict(scaler.transform(X_val)))
    print(f"Hold-out validation accuracy (80/20 split): {val_acc:.4f}")

    # Refit on the full training set for final predictions
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(max_iter=1000, random_state=SEED)
    model.fit(scaler.transform(X), y)
    print("Refit model on all training rows. Coefficients (standardized):")
    coefs = pd.Series(model.coef_[0], index=FEATURES).sort_values()
    print(coefs.round(3).to_string())

    step(5, "Accuracy on the training set")
    train_acc = accuracy_score(y, model.predict(scaler.transform(X)))
    print(f"Training accuracy: {train_acc:.4f}")

    step(6, "Load test data and predict")
    test = load_csv("test.csv")
    test = add_title(test)
    test = transform(test, fills, "test.csv")  # uses TRAIN fill values
    preds = model.predict(scaler.transform(test[FEATURES].values))
    out = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": preds})
    out_path = os.path.join(DATA_DIR, "predictions_python.csv")
    out.to_csv(out_path, index=False)
    print(f"\nPredicted survival rate on test set: {preds.mean():.1%}")
    print(f"Saved {len(out)} predictions to {out_path}")

    step(7, "Accuracy on the test set")
    if "Survived" in test.columns:
        print(f"Test accuracy: {accuracy_score(test['Survived'], preds):.4f}")
    else:
        ref = load_csv("gender_submission.csv", required=False)
        if ref is None:
            print("test.csv has no labels and gender_submission.csv not found; "
                  "submit predictions_python.csv to Kaggle for the true test score.")
        else:
            merged = out.merge(ref, on="PassengerId", suffixes=("", "_ref"))
            acc = accuracy_score(merged["Survived_ref"], merged["Survived"])
            print("Kaggle does not publish test labels. Comparing against "
                  "gender_submission.csv (Kaggle's 'all women survive' baseline) "
                  "as a reference label set:")
            print(f"Test accuracy vs. gender_submission.csv: {acc:.4f}")
            print("For the official test score, upload predictions_python.csv to Kaggle.")

    print("\nDone.")


if __name__ == "__main__":
    np.random.seed(SEED)
    main()