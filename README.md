# Tennis Match Prediction

Predicting the winner of professional men's (ATP) tennis matches from features
engineered out of historical match records, and benchmarking a wide range of
classifiers, including a stacking ensemble, as described in `paper.txt`.

## Repository layout

| File | Purpose |
|------|---------|
| `import-dataset.py` | Downloads the ATP dataset from Kaggle via `kagglehub` (`dissfya/atp-tennis-2000-2023daily-pull`). |
| `atp_tennis.csv` | Raw match data (git-ignored; place it in the project root). |
| `main1.ipynb` | Data cleaning, feature engineering, and classical ML model comparison. Writes `df_final.parquet`. |
| `main2.ipynb` | Neural models (PyTorch MLP, TabNet) on the engineered features. |
| `df_final.parquet` | Output of `main1.ipynb` with engineered features (git-ignored). |
| `requirements.txt` | Pinned Python dependencies. |
| `.python-version` | Python version used (3.12). |

## Requirements

- **Python 3.12** (notebooks were run on 3.12.11 / 3.12.12; 3.10+ is likely fine but untested)
- Packages listed in `requirements.txt`
- A GPU is not required. `torch` pulls in large CUDA wheels on Linux; for a
  CPU-only install use `pip install torch --index-url https://download.pytorch.org/whl/cpu`
  before installing the rest.

## Setup

```bash
python3.12 -m venv tennis-env
source tennis-env/bin/activate
pip install -r requirements.txt
```

## Getting the data

Either download `atp_tennis.csv` manually from
<https://www.kaggle.com/datasets/dissfya/atp-tennis-2000-2023daily-pull>
and put it in the project root, or run `python import-dataset.py` (set
`file_path` in the script to the CSV name; Kaggle credentials may be required).

The raw columns are: Tournament, Date, Series, Court, Surface, Round, Best of,
Player_1, Player_2, Winner, Rank_1, Rank_2, Pts_1, Pts_2, Odd_1, Odd_2, Score.

## Usage

Run the notebooks in order (`jupyter lab` or open in VS Code):

1. **`main1.ipynb`**
   - Cleans the data (null check, `Date` to datetime, chronological sort) and derives `Loser`.
   - Builds leak-free, *pre-match* features by walking matches in date order:
     - **Elo** (global and surface-specific; initial rating 1000, K = 32) -> `elo_diff`, `elo_surf_diff`
     - **Head-to-head** record difference -> `h2h_pre`
     - **Recent form**, wins in each player's last 10 matches -> `recent_form_diff`
   - Saves the result to `df_final.parquet`.
   - Splits by time: matches before `2022-01-01` train, the rest test.
     Each match becomes two rows (winner perspective, label 1; loser perspective,
     label 0, features negated) so classes are balanced.
   - Evaluates linear models, SVMs, trees/ensembles, kNN, Naive Bayes, discriminant
     analysis, MLP, LightGBM, XGBoost, voting, bagging, boosting, and a
     **stacking ensemble** (LogReg, SVC, XGBoost, GradientBoosting -> LogReg meta-learner).
2. **`main2.ipynb`**: loads `df_final.parquet` and trains a PyTorch MLP and TabNet.
   It expects `X_train`/`X_test` from the same split as `main1.ipynb`; run that
   split first or copy the split cells over.

## Notes

- Features use only information available before each match, and the split is
  chronological, to avoid look-ahead leakage.
- The Elo/H2H/form loops use `iterrows()` and are slow on the full dataset
  (a few minutes). SVC and stacking are the slowest model cells.
- `paper.txt` (git-ignored) holds the draft write-up of the approach.
