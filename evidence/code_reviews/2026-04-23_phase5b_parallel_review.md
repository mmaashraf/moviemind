# Periodic Code Review - Phase 5b Parallel Check

## Scope Reviewed
- `src/tune_dl.py`
- Running terminal output for ongoing 50-trial Optuna run

## Findings
1. **Repeated CSV loading inside each trial**
   - Evidence: old `objective()` read train/test CSV every trial.
   - Impact: unnecessary overhead and slower tuning.
   - Fix applied: data is now initialized once in `initialize_tuning_data()` and reused by all trials.

2. **Dtype warning noise from full CSV reads**
   - Evidence: terminal shows repeated `DtypeWarning` about mixed types.
   - Impact: noisy logs and possible parse overhead.
   - Fix applied: use `usecols=REQUIRED_COLS` and `low_memory=False` while loading only needed numeric columns.

3. **Evidence capture was fragmented**
   - Evidence: tuning output mainly in console/model files.
   - Impact: harder to audit phase history.
   - Fix applied: mirrored log writes to `evidence/phase5b/tune_dl_observations.txt`.

## Notes
- Current running job started before these optimizations, so it will finish with old behavior.
- Next run will automatically use the optimized and cleaner logging flow.
