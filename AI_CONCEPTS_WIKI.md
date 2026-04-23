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

### Time-Based Split (Preventing Data Leakage)
In time-series or sequential data, you cannot randomly split 80% train / 20% test. If you do, the model might learn from a rating made in 2003 to predict a rating made in 2000. This is "Data Leakage" (predicting the past using the future). A time-based split sorts data chronologically and trains on the past to predict the future.

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
