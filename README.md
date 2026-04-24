# MovieMind: Movie Recommendation System & AI Agent

This repository contains an end-to-end Machine Learning and Deep Learning pipeline for a Movie Recommendation System, built on the classic MovieLens 1M dataset. 

It is designed as an academic Capstone Project following a strict AI Development Life Cycle (AIDLC).

## Architecture & Features
*   **Data:** MovieLens 1M (includes user demographics like Age, Gender, Occupation).
*   **Machine Learning:** Linear Regression, Random Forest, Gradient Boosting (Content-Based).
*   **Deep Learning:** PyTorch Neural Collaborative Filtering (NCF) with User/Movie Embeddings (Hybrid).
*   **Explainable AI (XAI):** Models are designed to output reasoning behind recommendations.
*   **NLP Agent:** (Upcoming) A natural language interface to query recommendations.

---

## 🚀 How to Run the Project

Follow these steps in order to reproduce the pipeline from raw data to trained models.

### Step 0: Environment Setup
Ensure you have Python 3 installed. We recommend creating a virtual environment.
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 1: Data Acquisition
Download and extract the MovieLens 1M dataset into the `data/` folder.
```bash
python3 src/data_loader.py
```

### Step 2: Exploratory Data Analysis (EDA)
Explore the data distributions, matrix sparsity, and the "Cold Start" problem.
*   Open and run `notebooks/01_eda.ipynb`
*   Open and run `notebooks/02_long_tail_and_cold_start.ipynb`

### Step 3: Feature Engineering
Generate statistical features and perform a strict Time-Based Train/Test split to prevent data leakage.
```bash
python3 src/features.py
```
*(This generates `train_features.csv` and `test_features.csv` in `data/processed/`)*

### Step 4: Machine Learning Models
Train traditional ML baseline models to establish a performance benchmark.
```bash
python3 src/ml_models.py
```
*(This trains the models and saves `.pkl` files to the `models/` directory)*

### Step 5: Deep Learning Model
Train the Neural Collaborative Filtering model using PyTorch (Hardware accelerated via `mps` on Apple Silicon).
```bash
python3 src/dl_model.py
```
*(This trains for 10 epochs, saves `ncf_model.pt`, and generates a loss curve plot)*

---
*Note: This README acts as the central execution wiki and will be updated as new phases (API, UI, Agent) are completed.*