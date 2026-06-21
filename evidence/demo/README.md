# MovieMind demo videos

Automated walkthroughs recorded with **Playwright** (Chromium).

## Output (`video/`)

| File | Description |
|------|-------------|
| `video/moviemind_demo.webm` | Web app: all manual models + output review + diversity + Local LLM + tabs |
| `video/moviemind_capstone_notebook.webm` | Capstone notebook: cell-by-cell execution + outputs |
| `video/backup/` | Previous recordings (e.g. `moviemind_demo_v1_2026-06-21.webm`) |

## Record web app demo

```bash
source .venv/bin/activate
pip install -r requirements-demo.txt && playwright install chromium
bash scripts/record_demo_video.sh
```

## Record capstone notebook demo

```bash
bash scripts/record_notebook_video.sh
```

Uses fast path (`MOVIEMIND_SKIP_TUNE_*=1`) when artifacts exist. Options: `--pause-ms`, `--help`.

## Web app flow

1. **Recommend · Manual** — user 1161, diversity 0.35, every model + **output review**
2. Diversity impact + taste map
3. **Agent (NLP)** — Local LLM **Parse Query** + **Multi-step tool agent** (both with output review)
4. Each result review opens **Taste Map** expander + diversity panels

Requires `data/` + `models/` and Ollama (`llama3.1:8b`) for Local LLM.
