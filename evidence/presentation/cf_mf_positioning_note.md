# Collaborative Filtering vs Matrix Factorization in MovieMind

## Objective Positioning (for slide use)

In this project, collaborative filtering is implemented through an embedding-based Neural Collaborative Filtering (NCF) model, which learns latent user and movie vectors from interaction data.  
This captures the core matrix-factorization intuition (users/items in shared latent space), but we did not implement a separate classical matrix-factorization pipeline (such as SVD/ALS) as an independent baseline model.

---

## What we implemented

- **Collaborative Filtering approach used:** Neural Collaborative Filtering (PyTorch)
- **Where in code:** `src/dl_model.py`
- **Mechanism:**
  - `user_embedding`
  - `movie_embedding`
  - dense interaction layers on top of latent vectors + engineered features

---

## What we did not implement (explicitly)

- No standalone classical MF module:
  - no SVD baseline
  - no ALS solver
  - no explicit `R ≈ U * V^T` training script tracked as a separate model family

---

## Why this is still valid for objective

- The NCF embedding layer is a neural extension of latent-factor collaborative filtering.
- It preserves CF intuition while allowing nonlinear interaction modeling.
- For this capstone scope, this satisfies the collaborative-filtering objective, while keeping room for future addition of classical MF baselines.

---

## One-line version (for quick defense)

"We implemented collaborative filtering via embedding-based NCF (latent-factor CF), but we did not include a separate classical matrix-factorization baseline like ALS/SVD in this version."

---

## Optional Future Work line

"Future improvement: add explicit MF baselines (SVD/ALS) for apples-to-apples comparison with neural CF."
