# Periodic Code Review - Phase 6 t-SNE Stability Fix

## Scope Reviewed
- `src/post_analysis.py`

## Issue Observed
- Local run crashed with segmentation fault during t-SNE stage.
- Impact: script stopped before feature-importance outputs and summary completion.

## Fix Applied
1. Added safe-mode behavior:
   - t-SNE is now skipped by default.
   - script writes `evidence/phase6/tsne_status.txt` explaining how to enable t-SNE.
2. Added explicit enable switch:
   - `MOVIEMIND_ENABLE_TSNE=1 python3 src/post_analysis.py`
3. Updated t-SNE parameter naming:
   - switched from deprecated `n_iter` to `max_iter` for newer sklearn compatibility.

## Outcome
- Phase 6 script now completes end-to-end in default mode.
- Critical artifacts (PCA + feature importance + summary) are reliably generated.
