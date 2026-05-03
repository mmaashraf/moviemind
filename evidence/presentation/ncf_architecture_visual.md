# NCF Architecture Visual (Raw vs Tuned)

## 1) Raw NCF (from `src/dl_model.py`)

```mermaid
flowchart LR
    U[User ID] --> UE[User Embedding<br/>dim=32]
    M[Movie ID] --> ME[Movie Embedding<br/>dim=32]
    D[Dense Features<br/>6 values] --> C
    UE --> C[Concatenate]
    ME --> C
    C --> L1[Linear 70 -> 128<br/>ReLU + Dropout(0.2)]
    L1 --> L2[Linear 128 -> 64<br/>ReLU + Dropout(0.2)]
    L2 --> L3[Linear 64 -> 32<br/>ReLU]
    L3 --> O[Linear 32 -> 1<br/>Predicted Rating]
```

**Shape intuition**
- Embeddings: `32 + 32 = 64`
- Dense features: `6`
- Input to first layer: `64 + 6 = 70`
- Hidden stack: `128 -> 64 -> 32`
- Output: `1`

---

## 2) Tuned NCF (best Optuna trial)

```mermaid
flowchart LR
    U2[User ID] --> UE2[User Embedding<br/>dim=32]
    M2[Movie ID] --> ME2[Movie Embedding<br/>dim=32]
    D2[Dense Features<br/>6 values] --> C2
    UE2 --> C2[Concatenate]
    ME2 --> C2
    C2 --> T1[Linear 70 -> 256<br/>ReLU + Dropout(0.4431)]
    T1 --> T2[Linear 256 -> 64<br/>ReLU + Dropout(0.4431)]
    T2 --> T3[Linear 64 -> 32<br/>ReLU + Dropout(0.4431)]
    T3 --> O2[Linear 32 -> 1<br/>Predicted Rating]
```

**Best tuned params (from `models/best_dl_params.txt`)**
- `embedding_dim = 32`
- `n_layers = 3`
- `n_units_l0 = 256`
- `n_units_l1 = 64`
- `n_units_l2 = 32`
- `dropout_rate = 0.4431`
- `learning_rate = 0.00544`

---

## 3) Quick comparison

- Raw: `70 -> 128 -> 64 -> 32 -> 1`
- Tuned: `70 -> 256 -> 64 -> 32 -> 1`
- Biggest structural change: first hidden layer widened from `128` to `256`, and dropout increased from `0.2` to `~0.443`.
