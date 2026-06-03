# Security review (pre-public repository)

MovieMind is **for local run on your machine right now** — a course demo and reviewer workflow, **not** a deployed internet service.

**Reviewed:** 2026-06-03 · **Scope:** `src/`, `app/`, `scripts/`, tracked files (no secrets in git history for `ghp_`, `api_key`, etc.)

---

## Local-only scope (read this first)

This project **does not implement** and **does not target**:

| Capability | Status |
|------------|--------|
| **TLS / HTTPS** | Not included — HTTP on `127.0.0.1` only |
| **API authentication** | Not included — no API keys, sessions, or OAuth |
| **Authorization / RBAC** | Not included |
| **Rate limiting** | Not included |
| **Production hardening** | Out of scope unless you fork and extend the project |

**Intended use today:** one developer or reviewer on **localhost**:

- API: `http://127.0.0.1:8000` (see `scripts/restart_moviemind.sh`)
- UI: `http://127.0.0.1:8502`
- Ollama (optional): `http://127.0.0.1:11434`

Do **not** expose these ports on `0.0.0.0` or the public internet and treat the stack as production-ready. It is a **local capstone demo** only.

---

## Verdict: OK to make the repo public

No committed API keys, passwords, or `.env` files were found. `.gitignore` excludes `data/`, `models/`, `.env`, and `moviemind-artifacts.tar.gz`.

Making the repo **public** removes the need for reviewer GitHub tokens (§3 in `REVIEWER_SETUP.md`). The **release tarball** can be downloaded without authentication.

Publishing the **source repo** is fine; running the **servers on the open internet** without building TLS, auth, and rate limits yourself is not supported by this project.

---

## What becomes public

| Item | Risk level | Notes |
|------|------------|--------|
| Source code | Low | Intended |
| `evidence/` logs and JSON | Low | Smoke outputs; no secrets observed |
| Git history | Low | No PATs found via history search |
| **Release asset** (`moviemind-artifacts.tar.gz`) | Low | MovieLens 1M (public dataset); user ids are anonymous numeric ids |
| Your GitHub username / repo name | Low | Already visible if collaborators existed |

**Not in git (good):** raw `data/`, `models/`, local `.env`, 47MB tarball in repo root.

---

## Findings (by area)

### Secrets and credentials

- `.env` is gitignored; no `.env` tracked.
- Docs mention `MOVIEMIND_GITHUB_TOKEN` as **examples only** — never commit real tokens.
- `MOVIEMIND_API_LLM_ENABLED` gates cloud LLM; no API client secrets in repo when disabled (default).

### API (`src/api/app.py`)

| Topic | Status | Guidance |
|-------|--------|----------|
| Authentication | None | **By design** for local demo. Bind `127.0.0.1` only (`restart_moviemind.sh` default). |
| Authorization | None | Anyone who can reach the API can call `/recommend`, `/agent/query`, etc. |
| Input validation | Partial | Pydantic bounds on `user_id`, `top_n`, `max_turns`; genre guardrails in NLP/agent tools. |
| User enumeration | Low | `/users/{id}/summary` returns 404 out of range — acceptable for MovieLens demo. |
| OpenAPI `/docs` | Exposed | Fine locally; disable or protect if ever deployed publicly. |

### LLM / Ollama (`nlp.py`, `agent_loop.py`)

| Topic | Status | Guidance |
|-------|--------|----------|
| Prompt injection | Inherent | User query is sent to Ollama. Acceptable for local capstone; do not expose agent to untrusted internet users. |
| SSRF via `MOVIEMIND_OLLAMA_URL` | Env-only | Server-side env var, not client-controlled. Keep Ollama on localhost. |
| Cost / abuse | N/A on localhost | Not applicable — no rate limits in this repo |

### Pickle / joblib models (`model_registry.py`)

- Models load from **local files** you trained or downloaded from **your** release.
- Standard sklearn risk: **only load pickles you trust** (your own artifacts).

### Streamlit (`app/streamlit_app.py`)

- Reads markdown under project root for evidence tab — paths are fixed in code, not user-supplied.
- Calls API at `MOVIEMIND_API_URL` (default localhost).

### Scripts

- `download_review_artifacts.sh` — optional `Authorization` header from env; token not logged.
- `restart_moviemind.sh` — `uvicorn --reload` on by default: fine for dev, off for any shared server.

---

## Before toggling repo to Public (checklist)

- [ ] Confirm no `.env` or `*.pem` staged: `git status`
- [ ] Confirm `moviemind-artifacts.tar.gz` is not committed (in `.gitignore`)
- [ ] GitHub → Settings → ensure no accidental secret scanning alerts after push
- [ ] Update course instructions: public clone URL, no PAT for artifacts
- [ ] Optional: GitHub → Settings → Disable “Allow fork” if you care about copies (not security-critical)

## Safe run defaults (reviewers and you)

```bash
# API and UI on localhost only (defaults in restart_moviemind.sh)
MOVIEMIND_API_HOST=127.0.0.1
# Do not bind 0.0.0.0 on a shared or cloud VM without a firewall and auth
```

If you later deploy beyond localhost, you must add your own HTTPS, authentication, rate limiting, and network controls. **That is not part of MovieMind as shipped.**

---

## If the repo stays private

Reviewers need collaborator access **and** `MOVIEMIND_GITHUB_TOKEN` for release download — see `REVIEWER_SETUP.md` §3.

## If the repo is public

- Clone via HTTPS/SSH without special repo permissions.
- `download_review_artifacts.sh` works **without** `MOVIEMIND_GITHUB_TOKEN`.
- Same localhost security assumptions apply.
