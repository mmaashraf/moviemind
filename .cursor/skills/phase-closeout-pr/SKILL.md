---
name: phase-closeout-pr
description: Prepare phase completion git hygiene and PR flow for MovieMind. Use when the user finishes a phase and asks to commit, push, open PRs, sync progress docs, or create phase-specific PRs.
---

# Phase Closeout PR Workflow

Use this workflow when a MovieMind phase is completed.

## Scope

- clean branch and commit hygiene
- phase-specific docs sync
- push branch
- create/open PR links (single phase or all)

## Required Checks (in order)

1. Confirm current branch and status.
2. Ensure evidence artifacts for the phase exist under `evidence/<phase>/`.
3. Ensure docs are synced:
   - `docs/internal/PROGRESS_TRACKER.md`
   - `docs/internal/CONTEXT_HANDOVER.md`
   - `README.md`
   - `docs/internal/learning/AI_CONCEPTS_WIKI.md` (when new concepts were learned)
4. Make phase-focused commits (avoid mixed commit scope).
5. Push branch to origin.
6. Create/open PR using helper script.

## PR Helper Commands

Run from repo root:

```bash
# single phase PR
bash evidence/pr/create_prs.sh --phase 6

# all phase PRs
bash evidence/pr/create_prs.sh --phase all
```

Valid phase filters:
- `1`
- `2`
- `3`
- `4`
- `6`
- `all`

## Output Format to User

When done, report:
- current branch
- commit hashes created (if any)
- push result
- PR URL(s) opened/created
- any blocker (`gh` missing, auth missing, branch missing)

## Notes

- If `gh` is unavailable, script prints compare URLs and may open browser tabs.
- Keep commits phase-based and concise.
- Never push to `main` directly.
