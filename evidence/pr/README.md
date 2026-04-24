# PR Creation Kit

This folder stores reusable PR metadata and a helper script.

## Files

- `pr_plan.md`: all recommended PRs with base/head, title, and body.
- `create_prs.sh`: helper script to open/create PRs quickly.

## Usage

From repo root:

```bash
bash evidence/pr/create_prs.sh
```

Single phase:

```bash
bash evidence/pr/create_prs.sh --phase 6
```

### Behavior

- If `gh` CLI is available and authenticated:
  - it can create PRs directly.
- If `gh` is not available:
  - it prints/open compare URLs so you can create PRs in browser quickly.

## Notes

- Review each PR before submission.
- If a PR already exists for the same head/base, skip that entry.
