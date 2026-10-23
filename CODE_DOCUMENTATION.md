# Code Documentation

Detailed reference for the code in this repository. For setup and a quick
overview, see [README.md](README.md).

## Contents

1. [Pipeline overview](#pipeline-overview)
2. [`import-dataset.py`](#import-datasetpy)
3. [`main1.ipynb`](#main1ipynb)
4. [`main2.ipynb`](#main2ipynb)
5. [Data reference](#data-reference)
6. [Design decisions and caveats](#design-decisions-and-caveats)
7. [Extending the project](#extending-the-project)

## Pipeline overview

```
atp_tennis.csv --main1.ipynb--> cleaning -> Elo / H2H / form features -> df_final.parquet
                                                   |
                                                   v
                                     time split + two-rows-per-match
                                                   |
                         +-------------------------+-------------------------+
                         v                                                   v
              main1.ipynb: classical ML,                       main2.ipynb: PyTorch MLP
              voting / stacking / bagging / boosting           and TabNet
```

## `import-dataset.py`

Downloads the Kaggle dataset `dissfya/atp-tennis-2000-2023daily-pull` using
`kagglehub.load_dataset` with the pandas adapter and prints the first five
rows. `file_path` is empty and must be set to the CSV name inside the dataset
before it will load a file. The script does not save anything; the project
already includes `atp_tennis.csv`, so it is only needed to refresh the data.

## `main1.ipynb`

### 1. Loading and cleaning

| Step | Code | Notes |
|------|------|-------|
| Load | `pd.read_csv('atp_tennis.csv')` | One row per match. |
| Null check | `df.isnull().sum()` | The dataset has no nulls. |
| Dates | `pd.to_datetime(df['Date'])` | Needed for ordering and splitting. |
| Sort | `sort_values("Date").reset_index(drop=True)` | **Required**: all features are computed by walking matches in chronological order. |
| Loser | `np.where(Winner == Player_1, Player_2, Player_1)` | Adds a `Loser` column. |

### 2. Feature engineering

Every function takes the date-sorted DataFrame, returns a copy with new
columns, and iterates row by row. The value stored for a match is always the
state **before** the match, and the state is updated **after** it, so no
feature sees its own match's result.

#### `compute_elo(df, k=32, init_elo=1000, surface_col="Surface")`

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `k` | 32 | Update step size. |
| `init_elo` | 1000 | Starting rating for unseen players (and unseen player-surface pairs). |
| `surface_col` | `"Surface"` | Column used for the surface-specific rating. |

Keeps one global rating and one rating per (player, surface). For a match with
winner rating `w` and loser rating `l`:

```
expected_w = 1 / (1 + 10 ** ((l - w) / 400))
winner += k * (1 - expected_w)
loser  += k * (0 - (1 - expected_w))
```

The same update is applied to the surface ratings, using surface-specific
expectations. Added columns:

| Column | Description |
|--------|-------------|
| `winner_elo_pre`, `loser_elo_pre` | Global ratings before the match. |
| `elo_diff` | `winner_elo_pre - loser_elo_pre`. |
| `winner_elo_surf_pre`, `loser_elo_surf_pre` | Surface ratings before the match. |
| `elo_surf_diff` | Difference of the surface ratings. |

#### `add_h2h_feature(df)`

Keeps a nested counter `h2h[a][b]` = number of times `a` has beaten `b`.
Before each match it records `h2h[winner][loser] - h2h[loser][winner]`
(positive means the eventual winner led the head-to-head), then increments
`h2h[winner][loser]`. Added column: `h2h_pre`.

#### `add_recent_form(df, N=10)`

Keeps a `deque(maxlen=N)` per player holding 1 (win) / 0 (loss) for their last
`N` matches. Before each match it records
`wins_in_last_N(winner) - wins_in_last_N(loser)`, then appends the result to both
players' deques. Added column: `recent_form_diff`. Players with fewer than `N`
matches simply have shorter windows.

The finished DataFrame is saved with
`df.to_parquet('df_final.parquet', engine='fastparquet')`.

### 3. Train/test split and `make_training_rows(df)`

- `cutoff_date = "2022-01-01"`: matches before it are training data, the rest are
  test data. In `main1.ipynb` training uses all earlier matches.
- The engineered columns are expressed from the winner's point of view, so every
  raw row has label 1. To get a balanced problem, `make_training_rows` emits
  **two rows per match**:

| Row | `elo_diff`, `elo_surf_diff` | `h2h_pre`, `recent_form_diff` | `label` |
|-----|-----------------------------|-------------------------------|---------|
| Winner's view | as stored | as stored | 1 |
| Loser's view | negated | negated | 0 |

Output feature columns: `elo_diff`, `elo_surf_diff`, `h2h_pre`,
`recent_form_diff`. `X_*` holds those four; `y_*` holds `label`.

### 4. Models evaluated

Each model section follows the same pattern: fit on `X_train, y_train`, predict
on `X_test`, print accuracy and a `classification_report`.

| Family | Models |
|--------|--------|
| Linear | LogisticRegression, SGDClassifier (scaled), RidgeClassifier, Perceptron, PassiveAggressiveClassifier |
| SVM | SVC, LinearSVC, NuSVC (all with `StandardScaler`) |
| Trees / ensembles | DecisionTree, Bagged ExtraTree, RandomForest, GradientBoosting, HistGradientBoosting, AdaBoost |
| Neighbours | KNeighborsClassifier (k=5) |
| Naive Bayes | GaussianNB, MultinomialNB (on `abs(X)`), BernoulliNB |
| Discriminant analysis | LDA, QDA |
| Neural | sklearn `MLPClassifier` |
| Boosting libraries | LightGBM `LGBMClassifier`, `XGBClassifier` |
| Ensembles | Soft `VotingClassifier`, two `StackingClassifier`s, Bagging, AdaBoost, GradientBoosting, XGBoost |

**Stacking (main configuration)**: base learners LogisticRegression, SVC
(`probability=True`), XGBoost, GradientBoosting; meta-learner LogisticRegression.
An earlier experiment stacks RandomForest + LinearSVC + GradientBoosting.

`MultinomialNB` needs non-negative input, hence `np.abs(X_train)`.

## `main2.ipynb`

Loads `df_final.parquet` and does not depend on `main1.ipynb`'s in-memory state.

1. **Encoding**: every `object` column is converted with `OrdinalEncoder`
   (one encoder is refit per column). This only affects `df.info()` here; the
   model inputs are the four engineered features.
2. **Split**: same `cutoff_date = "2022-01-01"` for test, but training is limited
   to matches from `begin_date = "2018-01-01"` onward, so the training set is
   smaller than in `main1.ipynb`. Results are therefore not directly comparable.
3. **`make_training_rows`**: identical to `main1.ipynb`.
4. **`TabularDataset`**: `torch.utils.data.Dataset` over float32 features and
   int64 labels; `DataLoader` with `batch_size=64`, shuffled for training.
5. **`MLP(input_dim, hidden_dim1, hidden_dim2, output_dim)`**:
   `Linear(in,256) - ReLU - Linear(256,64) - ReLU - Linear(64,32) - ReLU - Linear(32,2)`.
   Trained for 30 epochs with Adam (lr 1e-3) and cross-entropy loss. The printed
   per-epoch loss is that of the last batch only, not an epoch average.
6. **TabNet**: `TabNetClassifier(n_steps=3, mask_type='sparsemax')`, Adam
   (lr 2e-2), StepLR (step 50, gamma 0.9), up to 200 epochs, patience 15. The test
   set is used as the `eval_set` for early stopping (see caveats).

## Data reference

### Raw columns (`atp_tennis.csv`)

`Tournament`, `Date`, `Series`, `Court`, `Surface`, `Round`, `Best of`,
`Player_1`, `Player_2`, `Winner`, `Rank_1`, `Rank_2`, `Pts_1`, `Pts_2`,
`Odd_1`, `Odd_2`, `Score`.

### Added columns (`df_final.parquet`)

`Loser`, `winner_elo_pre`, `loser_elo_pre`, `elo_diff`, `winner_elo_surf_pre`,
`loser_elo_surf_pre`, `elo_surf_diff`, `h2h_pre`, `recent_form_diff`.

## Design decisions and caveats

- **No look-ahead leakage in features**: values are read before the update, and
  the split is by date rather than random.
- **Mirrored test rows**: each test match appears twice with negated features and
  opposite labels. Accuracy is therefore symmetric by construction, and a model
  that predicts from the sign of `elo_diff` alone will already score well.
- **Elo is computed over the whole dataset before splitting**. Test-period
  ratings are built only from earlier matches, so there is no leakage, but ratings
  carry over from train into test, as they would in practice.
- **TabNet early stopping uses the test set**, which makes its reported test
  accuracy slightly optimistic. Use a validation slice of the training data for
  a clean estimate.
- **Unused columns**: ranking, points, and betting odds (`Rank_*`, `Pts_*`,
  `Odd_*`) are not used as features. Betting odds in particular are a strong
  predictor and a natural baseline to compare against.
- **Performance**: all feature functions and `make_training_rows` use
  `iterrows()`; vectorising or using `itertuples()` would be much faster.
- **Reproducibility**: most models set `random_state`, but several (e.g.
  `SVC`, `XGBClassifier`, `GradientBoostingClassifier` in the voting/stacking
  cells, and the PyTorch MLP) do not.

## Extending the project

- **New feature**: write a function with the same shape as `add_h2h_feature`
  (read state, append, then update), call it before saving the parquet, and add the
  feature (with a sign flip if it is antisymmetric) to both rows in
  `make_training_rows`.
- **New model**: add a cell that fits on `X_train`/`y_train` and reports
  `accuracy_score` and `classification_report` on `X_test`/`y_test`.
- **Different split**: change `cutoff_date` (and `begin_date` in `main2.ipynb`).
