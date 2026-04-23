# Live Monitor Notes - 2026-04-23

## Tuning Run Snapshot
- Command: `python src/tune_dl.py`
- Config: 50 trials, 3 epochs per trial
- Completed trial count: 50/50
- Final best RMSE: 1.0061 (trial 32)

## Observations
- Tuning completed successfully without crashes.
- Repeated `DtypeWarning` appeared because this run started before the data-loading fix.
- Best tuned DL result in this completed run is 1.0061, which is worse than the previously recorded 0.9853 (lower RMSE is better).

## Final Conclusion
- Tuned DL best score (1.0061) is still above Gradient Boosting benchmark (`0.897`).
- Gradient Boosting remains the winning production model at this stage.
