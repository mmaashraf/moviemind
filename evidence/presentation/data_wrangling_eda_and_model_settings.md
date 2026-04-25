# Data Wrangling, EDA, Train/Test, and DL Settings (Slide-Ready)

## 1) Data Wrangling - What was done

Implemented in `src/features.py`.

- Loaded MovieLens 1M raw files:
  - `ratings.dat` -> `userId, movieId, rating, timestamp`
  - `movies.dat` -> `movieId, title, genres`
  - `users.dat` -> `userId, gender, age, occupation, zipCode`
- Extracted `release_year` from movie title (fallback `1900` if missing).
- Built user-level behavioral features from ratings:
  - `user_rating_count`
  - `user_avg_rating`
- Built movie-level behavioral features:
  - `movie_rating_count`
  - `movie_avg_rating`
- Merged ratings + user stats + movie stats + demographics + movie metadata into one modeling table.
- Saved processed outputs to:
  - `data/processed/train_features.csv`
  - `data/processed/test_features.csv`

---

## 2) EDA - What we observed and why it mattered

From project notebooks and handover summary:

- User-item matrix is highly sparse (~95%).
  - Impact: memory-based CF/KNN alone is weak; embedding-style approaches are more suitable.
- Strong long-tail behavior in movies.
  - Impact: many items have few ratings; increases cold-start risk.
- Ratings are skewed positive (mean around 3.58).
  - Impact: predicting low ratings is harder; models can look optimistic.
- Demographic features available for all users.
  - Impact: useful side-information for hybrid modeling and richer explanations.

---

## 3) What is in Train vs Test data

Split logic is in `time_based_split()` in `src/features.py`.

- Data is sorted by `timestamp` first.
- Oldest ~80% -> **Train**
- Most recent ~20% -> **Test**
- This is a chronological split (not random) to avoid leakage.

### Columns present in processed train/test

- IDs/target/time:
  - `userId`, `movieId`, `rating`, `timestamp`
- Engineered user features:
  - `user_rating_count`, `user_avg_rating`
- Engineered movie features:
  - `movie_rating_count`, `movie_avg_rating`
- User metadata:
  - `gender`, `age`, `occupation`, `zipCode`
- Movie metadata:
  - `title`, `genres`, `release_year`

---

## 4) Activation and Loss Function used

From `src/dl_model.py` (NCF baseline):

- **Activation in hidden layers:** `ReLU`
- **Regularization:** `Dropout(0.2)` between dense layers
- **Output layer:** final linear layer with 1 unit (regression output)
- **Loss function:** `MSELoss` (mean squared error)
- **Reported metric during training:** `Val RMSE = sqrt(Val MSE)`
- **Optimizer:** `Adam`

### Baseline NCF architecture (dense block)
- `Linear -> ReLU -> Dropout`
- `Linear -> ReLU -> Dropout`
- `Linear -> ReLU`
- `Linear(32 -> 1)`

---

## 5) One-line defense summary

"We engineered user/movie statistical features, merged them with demographics and metadata, used a strict timestamp-based train/test split, and trained NCF with ReLU activations + MSE loss for rating prediction."
