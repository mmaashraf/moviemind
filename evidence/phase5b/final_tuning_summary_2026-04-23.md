# Phase 5b Final Tuning Summary (50 Trials)

## Run Details
- Script: `src/tune_dl.py`
- Trials: 50
- Epochs per trial: 3
- Completion status: success

## Final Best Result
- Best RMSE: **1.0061**
- Best hyperparameters:
  - `learning_rate`: 0.005440112043520528
  - `embedding_dim`: 32
  - `dropout_rate`: 0.4431288798539017
  - `n_layers`: 3
  - `n_units_l0`: 256
  - `n_units_l1`: 64
  - `n_units_l2`: 32

## Comparison vs Current Champion
- Gradient Boosting RMSE: **0.897**
- Tuned DL RMSE: **1.0061**
- Outcome: Gradient Boosting remains the winning model.

## Source Evidence
- `models/tune_dl_log.txt`
- `models/best_dl_params.txt`
- `evidence/phase5b/live_monitor_2026-04-23.md`
