# Titanic Survival Prediction in Docker (Python & R)

Predicts which passengers survived the Titanic (April 15, 1912) using **logistic regression**, packaged in two self-contained Docker containers: one in **Python** and one in **R**. Each container loads the Kaggle data, cleans and engineers features (printing every change to the terminal), trains a model, reports accuracy, and writes predictions for the test set.

> The dataset is **not** included in this repo. Step 2 below shows how to download it.

---

## Repository structure

```
mlds400-titanic/
├── .github/
│   └── CODEOWNERS            # repo owner must approve changes
├── .gitignore                # ignores src/data/*, .DS_Store, venvs, etc.
├── README.md
└── src/
    ├── data/                 # put the Kaggle CSVs here (git-ignored)
    │   └── .gitkeep
    ├── python/
    │   ├── Dockerfile
    │   ├── requirements.txt  # pinned package versions
    │   └── titanic.py
    └── r/
        ├── Dockerfile
        ├── install_packages.R
        └── titanic.R
```

---

## Prerequisites

| Tool | Why | Check it's installed |
|---|---|---|
| [Docker Desktop](https://www.docker.com/products/docker-desktop/) | builds and runs both containers | `docker --version` |
| [Git](https://git-scm.com/downloads) | clones this repo | `git --version` |
| A free [Kaggle](https://www.kaggle.com) account | downloads the data | |

**Make sure Docker Desktop is open and running** before Step 3 (the whale icon in your menu bar / system tray should be steady, not animating).

Python and R do **not** need to be installed on your machine. They run inside the containers.

---

## Step 1. Clone the repo

```bash
git clone https://github.com/shihyun-park/mlds400-titanic.git
cd mlds400-titanic
```

All remaining commands are run from this `mlds400-titanic/` folder (the repo root).

---

## Step 2. Download the data

1. Go to the competition data page: <https://www.kaggle.com/competitions/titanic/data>
2. Sign in, click **Join Competition** / accept the rules (required before downloading).
3. Click **Download All**. This gives you `titanic.zip`.
4. Unzip it and move the three CSV files into `src/data/`:

```
src/data/
├── train.csv               # 891 passengers, includes Survived
├── test.csv                # 418 passengers, no Survived column
└── gender_submission.csv   # Kaggle's example submission
```

From the terminal (macOS / Linux), assuming the zip is in your Downloads folder:

```bash
unzip ~/Downloads/titanic.zip -d src/data/
```

<details>
<summary>Alternative: download with the Kaggle CLI</summary>

```bash
pip install kaggle   # requires ~/.kaggle/kaggle.json API token (Kaggle → Settings → Create New Token)
kaggle competitions download -c titanic -p src/data
unzip src/data/titanic.zip -d src/data && rm src/data/titanic.zip
```
</details>

Verify:

```bash
ls src/data
# gender_submission.csv  test.csv  train.csv
```

---

## Step 3. Run the Python container

```bash
# Build the image (first build ~1 min)
docker build -t titanic-python src/python

# Run it, mounting the data folder into the container
docker run --rm -v "$(pwd)/src/data:/app/data" titanic-python
```

---

## Step 4. Run the R container

```bash
# Build the image (first build ~2-5 min while R packages install)
docker build -t titanic-r src/r

# Run it, mounting the data folder into the container
docker run --rm -v "$(pwd)/src/data:/app/data" titanic-r
```

### Windows users

Replace `$(pwd)` in the `docker run` commands:

| Shell | Command |
|---|---|
| PowerShell | `docker run --rm -v "${PWD}/src/data:/app/data" titanic-python` |
| Command Prompt | `docker run --rm -v "%cd%/src/data:/app/data" titanic-python` |

(Same for `titanic-r`.)

---

## What you should see

Each script prints 7 labeled steps:

| Step | What happens |
|---|---|
| 1. Load training data | reads `train.csv`, prints shape and first rows |
| 2. Explore | overall survival rate, missing values, survival by Sex and Pclass |
| 3. Feature engineering | every cleaning / feature step printed (how many values filled, new columns created) |
| 4. Train logistic regression | 80/20 hold-out validation accuracy, then refit on all training rows; prints coefficients |
| 5. Training accuracy | accuracy on `train.csv` |
| 6. Predict on test set | applies the **same** transformations (using training-set fill values) to `test.csv` and saves predictions |
| 7. Test accuracy | see note below |

Predictions are written back to your machine through the mounted folder:

- `src/data/predictions_python.csv`
- `src/data/predictions_r.csv`

Both are in Kaggle submission format (`PassengerId,Survived`).

### Note on test-set accuracy

Kaggle does **not** publish the true `Survived` labels for `test.csv`. So the scripts report:

- **Hold-out validation accuracy** (Step 4): the honest estimate of performance on unseen data.
- **Accuracy vs. `gender_submission.csv`** (Step 7): agreement with Kaggle's "all women survive, all men die" baseline. This is a reference, not ground truth.
- For the **official test accuracy**, upload `predictions_python.csv` or `predictions_r.csv` at <https://www.kaggle.com/competitions/titanic/submissions>.

---

## Models

Both use logistic regression with a 0.5 decision threshold. Missing values are imputed using statistics learned from the **training set only** (no leakage into test).

| | Python (`scikit-learn`) | R (`glm`, binomial) |
|---|---|---|
| Age imputation | median age by Title (Mr, Mrs, Miss, Master, Rare) | same |
| Features | Pclass, Sex, Age, SibSp, Parch, Fare, FamilySize, IsAlone, HasCabin, Embarked (one-hot), Title (one-hot) | Pclass (factor), Sex, Age, IsChild (<13), Fare, FamilySize, IsAlone, HasCabin, Embarked (factor) |
| Scaling | standardized (`StandardScaler`) | none |
| Why different | L2-regularized by default, so correlated Title/Sex dummies are fine | unpenalized `glm`: Title is left out to avoid quasi-separation with Sex, and FamilySize replaces SibSp/Parch to avoid perfect collinearity |

**Package versions.** Python: `python:3.11-slim`, numpy 2.1.3, pandas 2.2.3, scipy 1.14.1, scikit-learn 1.5.2 (`requirements.txt`). R: `rocker/r-ver:4.4.1` (CRAN pinned to that release's snapshot), readr, dplyr.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Cannot connect to the Docker daemon` | Open Docker Desktop and wait until it says *Running*. |
| `train.csv not found` | The CSVs aren't in `src/data/`, or you ran `docker run` from a folder other than the repo root. `cd` into `mlds400-titanic/` and check `ls src/data`. |
| `invalid reference format` on Windows | You used `$(pwd)` in PowerShell/CMD. Use the Windows commands above. |
| Want to rebuild from scratch | `docker build --no-cache -t titanic-python src/python` |
| Clean up images afterwards | `docker rmi titanic-python titanic-r` |

---

## Branching workflow

`feature/titanic-model` → PR → `develop` → PR → `master`. All code was written on the feature branch and merged through pull requests.