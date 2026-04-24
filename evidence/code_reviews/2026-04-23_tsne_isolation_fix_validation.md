# t-SNE Isolation Fix Validation - 2026-04-23

## Problem
- `MOVIEMIND_ENABLE_TSNE=1 python src/post_analysis.py` previously ended with segmentation fault.

## Fix
- Updated `src/post_analysis.py` to run t-SNE in a child Python process.
- If child crashes, parent process now survives and completes remaining Phase 6 artifacts.
- Added thread-limit env defaults in child process for extra stability.

## Validation Run
- Command executed:
  - `MOVIEMIND_ENABLE_TSNE=1 python src/post_analysis.py`
- Result:
  - Completed successfully (exit code `0`)
  - t-SNE outputs generated:
    - `evidence/phase6/user_embeddings_tsne_2d_sample.csv`
    - `evidence/phase6/user_embeddings_tsne_2d_sample.png`
  - Summary updated with `t-SNE status: completed`
