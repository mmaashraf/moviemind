# AI / ML / DL Knowledge Wiki

This wiki serves as a self-contained glossary and explanation of all Artificial Intelligence, Machine Learning, and Deep Learning concepts utilized in this Movie Recommendation System project. 

*This document will be updated as new techniques are implemented.*

---

## 1. Data & Recommendation Concepts

### The "Cold Start" Problem
A major challenge in recommendation systems. If a new movie is added (0 ratings) or a new user signs up (0 watch history), the system cannot reliably recommend things using traditional Collaborative Filtering because there is no historical data to "collaborate" with.

### The "Long Tail"
In any popularity-based dataset (movies, music, products), a tiny fraction of items (blockbusters) receives the vast majority of engagement (the "head"). The remaining 90% of items receive very little engagement, forming a massive "long tail" on a graph. Recommender systems must be designed to explore the long tail, otherwise, they will only recommend the same 10 blockbusters to everyone.

### Matrix Sparsity
If you create a grid of all Users (rows) vs. all Movies (columns), "Sparsity" is the percentage of empty cells where a user hasn't rated a movie. In our dataset, it is ~95%. High sparsity breaks simple algorithms like K-Nearest Neighbors (KNN), necessitating Matrix Factorization or Deep Learning.

### K-Nearest Neighbors (KNN) in Recommender Systems
KNN is a memory-based method that predicts by looking at the **K most similar neighbors**.

In recommenders, KNN appears in two common forms:
1. **User-based KNN:** Find users similar to a target user, then recommend movies those similar users liked.
2. **Item-based KNN:** Find movies similar to movies the user already liked, then recommend those similar movies.

**What does K mean?**
*   `K=5` means use the 5 nearest neighbors.
*   Small K can be noisy; large K can become too generic.

**Why KNN struggles in our dataset:**
Our matrix is highly sparse (~95%), so users and movies often have very little overlap. If overlap is weak, similarity scores become unreliable, and KNN recommendations degrade.

### Matrix Representation Used by KNN (What vectors are compared?)
In collaborative filtering KNN, we usually build a **User-Item rating matrix**:
*   Rows = `userId`
*   Columns = `movieId`
*   Cell value = `rating`
*   Empty cells = movie not rated by that user

Core columns used to construct this matrix are:
*   `userId`
*   `movieId`
*   `rating`
*   `timestamp` (only for time-based split, not for similarity math)

### Why vectors become very long
Yes, your intuition is correct. Vector lengths can be very large:
*   In **User-based KNN**, each user vector length is approximately number of movies.
*   In **Item-based KNN**, each movie vector length is approximately number of users.

So if users are ~6,040 and movies are ~3,883:
*   user vector can be length ~3,883
*   movie vector can be length ~6,040

Most entries are still empty due to sparsity, making these high-dimensional **sparse vectors**.

### Similarity as vector math
KNN compares these vectors using metrics like cosine or Pearson similarity.
That means recommendation quality depends on how much meaningful overlap exists between vectors.
With sparse overlap, similarity estimates become noisy, which is a key reason simple memory-based CF weakens in this project.

### Collaborative Filtering (Feynman Style)
Think of Collaborative Filtering as:
**"People who behaved like you in the past can help predict what you may like next."**

No deep movie metadata is required at first. The system mainly learns from interaction history (ratings/clicks).

#### Everyday analogy
Suppose:
*   You and 100 other people rate many movies.
*   A subset of those people has very similar taste to you.
*   They loved Movie X, but you have not watched it yet.

Collaborative Filtering says:
**"There is a strong chance you may also like Movie X."**

#### Two classic Collaborative Filtering styles
1. **User-User CF (Neighborhood):**
   Find users similar to target user, then borrow their preferences.

2. **Item-Item CF (Neighborhood):**
   Find items similar to items the user already liked.

Both are often implemented with KNN-style similarity search.

#### Matrix Factorization / Embedding-based CF (Modern style)
Instead of directly matching raw neighbors, we learn hidden factors:
*   user latent vector (taste profile)
*   item latent vector (movie characteristics in hidden space)

If user vector and movie vector align well, predicted rating is high.
This is the core intuition behind many modern recommender systems, including DL embedding models like NCF.

#### Why Collaborative Filtering is powerful
*   It captures crowd wisdom from real behavior.
*   It can recommend unexpected items outside simple genre rules.
*   It improves as interaction data grows.

#### Where it fails (important)
1. **Cold Start:** new users/items have little history.
2. **Sparsity:** too many missing ratings reduce reliable similarity.
3. **Popularity bias:** frequent items can dominate if not corrected.

#### How this maps to MovieMind
*   We explored neighborhood intuition and recognized sparsity/cold-start risk.
*   We used feature-rich ML and embedding-based DL to reduce dependence on raw overlap.
*   Phase 6 embedding analysis helps us inspect the learned collaborative space visually.

### Time-Based Split (Preventing Data Leakage)
In time-series or sequential data, you cannot randomly split 80% train / 20% test. If you do, the model might learn from a rating made in 2003 to predict a rating made in 2000. This is "Data Leakage" (predicting the past using the future). A time-based split sorts data chronologically and trains on the past to predict the future.

### Data Splitting: Train vs. Validation vs. Test
A rigorous ML project splits its data into three distinct sets:
1.  **Train Dataset (e.g., 70%):** The data the model actively learns from and updates its weights based upon.
2.  **Validation Dataset (e.g., 10%):** Used *during* development and hyperparameter tuning. The model does not learn from this, but the engineer uses the validation score to decide which architecture/hyperparameters are best.
3.  **Test Dataset (e.g., 20%):** Locked away in a vault until the very end of the project. It provides the final, unbiased metric of how the model will perform on completely unseen data in the real world.

---

## 2. Traditional Machine Learning (ML)

### Baseline Model
The simplest, most naive approach possible. For regression (predicting a number), the baseline is usually just "predict the average." It exists purely as a benchmark. If a complex AI cannot beat the naive baseline, the AI architecture is flawed.

### Random Forest (Bagging)
An ensemble algorithm. "Bagging" stands for Bootstrap Aggregating. The model builds many (e.g., 100) deep, complex Decision Trees independently and at the same time. Each tree looks at a random subset of the data. To make a final prediction, all 100 trees "vote" and the average is taken. This reduces overfitting and variance.

### Gradient Boosting
An ensemble algorithm based on "Boosting." Unlike Random Forest where trees are independent, Boosting builds trees **sequentially**. 
1. Tree 1 makes a prediction. It will have some errors (residuals).
2. Tree 2 is built specifically to predict and correct the errors of Tree 1.
3. Tree 3 corrects Tree 2, and so on.
It is a team of "weak learners" that slowly combine into one extremely strong learner.

### Hyperparameters: `max_depth` vs `n_estimators`
*   **`n_estimators`:** The total number of trees in the forest/ensemble.
*   **`max_depth`:** How many "splits" or questions a single tree can ask. It is **not** determined by the number of columns/features. It determines how complex the tree can get. A high `max_depth` on a dataset with lots of rows will cause severe overfitting (memorizing the data).

### Evaluation Metrics (RMSE & MAE)
*   **MAE (Mean Absolute Error):** The average distance between the predicted rating and the true rating. (e.g., an MAE of 0.7 means the model is off by 0.7 stars on average).
*   **RMSE (Root Mean Squared Error):** Similar to MAE, but because it squares the errors before averaging them, it heavily punishes predictions that are wildly wrong (e.g., guessing 1 star when the truth is 5 stars).

---

## 3. Deep Learning (DL)

### Neural Collaborative Filtering (NCF)
A deep learning architecture designed specifically for recommendation systems. It replaces traditional matrix math with neural network layers to learn non-linear interactions between users and items.

### Embeddings (`nn.Embedding`)
A technique to turn categorical data (like a User ID or Movie ID) into a dense vector of numbers (e.g., an array of 32 floats). Over time, the network updates these numbers so that similar movies end up with mathematically similar vectors. It is the core of how AI understands "meaning" and "taste."

### Dense / Fully Connected Layers (`nn.Linear`)
Standard neural network layers where every artificial neuron is connected to every neuron in the previous layer. Used to find complex mathematical patterns in the concatenated features.

### Activation Functions (ReLU)
**Rectified Linear Unit (ReLU)** is the most popular activation function. Without activation functions, a neural network is just a giant linear math equation. ReLU allows the network to learn non-linear, complex curves by outputting the input directly if it is positive, and outputting 0 if it is negative.

### Regularization (Dropout)
A technique to prevent overfitting. `Dropout(0.2)` randomly turns off 20% of the neurons in a layer during each training step. This forces the network to distribute its learning and prevents it from relying entirely on one dominant feature.

### Optimizers (Adam)
The algorithm that actually updates the weights of the network based on the errors. **Adam (Adaptive Moment Estimation)** is the industry standard because it dynamically calculates individual learning rates for every single parameter in the network, allowing for faster and more stable convergence.

### Epochs & Batch Size
*   **Epoch:** One complete pass through the entire training dataset.
*   **Batch Size:** Instead of updating the network after looking at 1 row, or waiting until it sees all 800,000 rows, we process the data in "batches" (e.g., 1024 rows at a time). This balances memory usage with training stability.

### Hyperparameter Tuning & Optuna
While the architecture (number of layers) and parameters (weights) are learned by the network, **Hyperparameters** are the settings configured *before* training begins (e.g., Learning Rate, Dropout percentage, Embedding Size). 
*   **Optuna:** A modern, define-by-run hyperparameter optimization framework. Instead of random guessing, it uses Bayesian optimization to intelligently search for the best combination of hyperparameters. It tracks previous experiments and focuses on the most promising settings to squeeze out maximum accuracy.

---

## 4. Post-Modeling Analysis & Explainable AI (XAI)

### Why Post-Modeling Analysis Matters
Training a model is not the end. A capstone-grade AI system must answer:
1. **How well does it perform?**
2. **Why is it predicting this?**
3. **What did it learn internally?**

Post-modeling analysis is where we make model behavior visible and explainable.

### Feature Importance (for Tree Models like Gradient Boosting)
Feature importance tells us which input features the model relied on most while making predictions.
For example, if `movie_avg_rating` has high importance, the model is heavily using movie quality/popularity signals.

This is useful because:
*   It improves trust (we can explain "why").
*   It helps debugging (we can spot if model relies on weak/spurious features).
*   It guides feature engineering for future improvements.

### Embedding Space Visualization
User embeddings are high-dimensional vectors (e.g., 32D). Humans cannot directly interpret 32 numbers per user.
So we project them to 2D for visualization.

The goal is to inspect whether users with similar tastes appear close together in the learned space.

### PCA (Principal Component Analysis)
PCA is a linear dimensionality reduction method.
It finds directions in the data with maximum variance and projects high-dimensional vectors into fewer dimensions (often 2D for plotting).

Strengths:
*   Fast and stable.
*   Good first visualization baseline.
*   Preserves broad/global structure reasonably well.

### t-SNE (t-distributed Stochastic Neighbor Embedding)
t-SNE is a nonlinear dimensionality reduction method focused on preserving **local neighborhoods**.
In simple words: points that are close in high-dimensional space are kept close in 2D.

Strengths:
*   Often reveals clusters more clearly than PCA.
*   Very useful for "who is similar to whom" intuition.

Limitations:
*   Slower than PCA.
*   Sensitive to settings (perplexity, iterations, random seed).
*   Can be unstable on some local environments, which is why our Phase 6 run uses safe mode by default and makes t-SNE optional.

### Safe Mode vs Optional t-SNE in This Project
In our implementation:
*   PCA is always generated (reliable core artifact).
*   t-SNE is optional and enabled only when requested.

Command to enable t-SNE explicitly:
```bash
MOVIEMIND_ENABLE_TSNE=1 python src/post_analysis.py
```

This gives us reproducibility and stability while still allowing deeper visualization when environment supports it.

---

## 5. API, Web App, and Agentic Layer Concepts

### FastAPI (Feynman Style)
Think of FastAPI as a clean "service counter" between UI and models.
Instead of UI code touching model files directly, UI sends structured requests like:
"Give me top 10 movies for user 15 using Gradient Boosting."

FastAPI then:
1. validates input schema,
2. chooses the model adapter,
3. returns a consistent JSON response.

This separation keeps the system modular and easier to test.

### Why Streamlit for this capstone?
Streamlit is the fastest way to build a serious AI demo interface in Python.
It is ideal for:
* model selection dropdowns,
* recommendation tables,
* explainability panels,
* experiment toggles (like runtime mode).

In short:
*FastAPI = backend contract + inference engine*
*Streamlit = interactive product experience*

### Model Registry (Why we need it)
A model registry is a single abstraction layer that hides model-specific differences.
Without it, each endpoint would need custom logic for baseline vs sklearn vs PyTorch.
With it, API routes call one unified interface:
* list models
* get model info
* predict
* recommend

This makes model switching in UI simple and robust.

### Model Inspector (Moderate Scope)
Model Inspector is an explainability and transparency page.
It answers:
* Which model is this?
* Is artifact available?
* What are key parameters?
* What metrics do we know?
* For GB: which features are most important?
* For DL: embedding size and parameter count.

This improves trust and helps debugging when results look surprising.

### Agentic Runtime Modes (Rule-only vs Local LLM vs API LLM)
Natural language queries can be interpreted in different ways:

1. **Rule-only**
   Deterministic parser. Best for reproducible benchmarking.

2. **Local LLM**
   Better language flexibility, runs locally, usually lower privacy risk.

3. **API LLM**
   Strongest language quality, but depends on external service and config.

### Why guardrails are mandatory for NLP parsing
Natural language is ambiguous.
If an LLM output is used directly without validation, it can produce unsupported keys, unsafe values, or unstable behavior.

So we enforce:
* schema validation,
* whitelist of supported filter keys,
* bounded numeric ranges (like top-N limits),
* deterministic fallback parser.

This keeps the agent helpful **and** reliable.
