#!/usr/bin/env python3
"""
Record a brief MovieMind UI demo video with Playwright.

Prerequisites:
  source .venv/bin/activate
  pip install playwright && playwright install chromium

Usage:
  python scripts/record_demo_video.py --manage-services --with-ollama
  python scripts/record_demo_video.py   # if API/UI already running
"""
from __future__ import annotations

import os
import argparse
import json
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "evidence" / "demo" / "video" / "moviemind_demo.webm"
UI_URL = "http://127.0.0.1:8502"
API_URL = "http://127.0.0.1:8000"


def _http_json(url: str, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def wait_for_api(max_sec: int = 120) -> None:
    deadline = time.time() + max_sec
    last_err = ""
    while time.time() < deadline:
        try:
            data = _http_json(f"{API_URL}/health", timeout=5)
            if data.get("status") in {"ok", "healthy"}:
                return
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            time.sleep(2)
    raise RuntimeError(f"API not healthy at {API_URL}/health: {last_err}")


def wait_for_ui(max_sec: int = 60) -> None:
    deadline = time.time() + max_sec
    last_err = ""
    while time.time() < deadline:
        try:
            req = urllib.request.Request(UI_URL, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            time.sleep(2)
    raise RuntimeError(f"UI not ready at {UI_URL}: {last_err}")


class ServiceManager:
    """Start/stop API + Streamlit for unattended demo capture."""

    def __init__(self, with_ollama: bool) -> None:
        self.with_ollama = with_ollama
        self._procs: list[subprocess.Popen] = []

    def start(self) -> None:
        if self.with_ollama:
            subprocess.run(
                ["bash", str(ROOT / "scripts" / "setup_local_ollama.sh")],
                cwd=ROOT,
                check=False,
            )
        env = {**os.environ, "MOVIEMIND_UVICORN_RELOAD": "0"}
        api_cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "src.api.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ]
        ui_cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app/streamlit_app.py",
            "--server.port",
            "8502",
            "--server.address",
            "127.0.0.1",
            "--server.headless",
            "true",
        ]
        log_dir = ROOT / "evidence" / "runtime"
        log_dir.mkdir(parents=True, exist_ok=True)
        api_log = open(log_dir / "demo_api.log", "w", encoding="utf-8")  # noqa: SIM115
        ui_log = open(log_dir / "demo_ui.log", "w", encoding="utf-8")  # noqa: SIM115
        self._procs.append(
            subprocess.Popen(
                api_cmd,
                cwd=ROOT,
                env=env,
                stdout=api_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        )
        self._procs.append(
            subprocess.Popen(
                ui_cmd,
                cwd=ROOT,
                env={**env, "MOVIEMIND_API_URL": API_URL},
                stdout=ui_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        )
        wait_for_api(120)
        wait_for_ui(90)

    def stop(self) -> None:
        for proc in self._procs:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        deadline = time.time() + 8
        for proc in self._procs:
            if proc.poll() is None and time.time() < deadline:
                try:
                    proc.wait(timeout=max(0, deadline - time.time()))
                except subprocess.TimeoutExpired:
                    proc.kill()
        self._procs.clear()


def fetch_model_labels() -> list[str]:
    payload = _http_json(f"{API_URL}/models")
    labels: list[str] = []
    for row in payload.get("models", []):
        if not row.get("available"):
            continue
        labels.append(f"{row['display_name']} ({row['model_id']})")
    if not labels:
        raise RuntimeError("No available models from GET /models")
    return labels


def show_banner(page, text: str) -> None:
    page.evaluate(
        """(text) => {
            let el = document.getElementById('moviemind-demo-banner');
            if (!el) {
                el = document.createElement('div');
                el.id = 'moviemind-demo-banner';
                el.style.cssText = [
                    'position:fixed','top:14px','left:50%','transform:translateX(-50%)',
                    'z-index:999999','background:#0f2744','color:#fff','padding:10px 22px',
                    'border-radius:8px','font:600 17px -apple-system,BlinkMacSystemFont,sans-serif',
                    'box-shadow:0 4px 18px rgba(0,0,0,.35)','pointer-events:none'
                ].join(';');
                document.body.appendChild(el);
            }
            el.textContent = text;
        }""",
        text,
    )


def wait_streamlit(page, ms: int = 900) -> None:
    page.wait_for_load_state("domcontentloaded")
    try:
        page.wait_for_load_state("networkidle", timeout=15000)
    except Exception:
        pass
    page.wait_for_timeout(ms)


def click_tab(page, name: str) -> None:
    page.get_by_role("tab", name=name, exact=True).click()
    wait_streamlit(page)


def expand_expander(page, title: str) -> None:
    summary = page.locator(f"details summary:has-text('{title}')")
    if summary.count() == 0:
        page.get_by_text(title, exact=False).first.click()
    else:
        details = summary.locator("xpath=ancestor::details[1]")
        if details.get_attribute("open") is None:
            summary.click()
    wait_streamlit(page, 500)


def select_labeled_option(page, label: str, option_substring: str) -> None:
    block = page.locator('[data-testid="stSelectbox"]').filter(has_text=re.compile(re.escape(label)))
    if block.count() == 0:
        block = page.locator('[data-testid="stSelectbox"]').filter(has=page.get_by_text(label, exact=True))
    target = block.first if block.count() else page.locator('[data-testid="stSelectbox"]').first
    target.locator('[data-baseweb="select"]').click()
    page.wait_for_timeout(400)
    page.locator('[role="option"]').filter(has_text=option_substring).first.click()
    wait_streamlit(page)


def set_user_id(page, user_id: int) -> None:
    inp = page.get_by_label("User ID")
    inp.click()
    inp.fill(str(user_id))
    inp.press("Tab")
    wait_streamlit(page)


def set_diversity(page, alpha: float) -> None:
    expand_expander(page, "Advanced Recommendation Controls")
    track = page.locator('[data-testid="stSlider"]').first
    box = track.bounding_box()
    if box:
        x = box["x"] + box["width"] * max(0.05, min(0.95, alpha))
        y = box["y"] + box["height"] / 2
        page.mouse.click(x, y)
    wait_streamlit(page)


def scroll_main(page, pixels: int = 500) -> None:
    page.evaluate(f"window.scrollBy(0, {pixels})")
    page.wait_for_timeout(600)


def scroll_to_top(page) -> None:
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(400)


def scroll_gradually(page, steps: int = 5, step_px: int = 380, pause_ms: int = 1100) -> None:
    """Slow scroll so tab content is readable in the recording."""
    for _ in range(steps):
        scroll_main(page, step_px)
        page.wait_for_timeout(pause_ms)


def _scroll_visible_dataframe(page, pause_ms: int, prefer_last: bool = False) -> bool:
    tables = page.locator('[data-testid="stDataFrame"]')
    count = tables.count()
    if count == 0:
        return False
    indices = range(count - 1, -1, -1) if prefer_last else range(count)
    for i in indices:
        table = tables.nth(i)
        try:
            if not table.is_visible():
                continue
            table.scroll_into_view_if_needed(timeout=5000)
            page.wait_for_timeout(pause_ms)
            scroll_gradually(page, steps=2, step_px=280, pause_ms=pause_ms // 2)
            return True
        except Exception:
            continue
    return False


def review_taste_map(page, pause_ms: int) -> None:
    show_banner(page, "Review · taste map (radar)")
    expand_expander(page, "Taste Map (Game-Style Radar)")
    page.wait_for_timeout(pause_ms)
    scroll_main(page, 200)
    page.wait_for_timeout(pause_ms // 3)


def review_recommendation_outputs(page, pause_ms: int) -> None:
    """Pause on recommendation table, diversity panels, and taste map after each run."""
    scroll_to_top(page)
    show_banner(page, "Review · recommendation table")
    _scroll_visible_dataframe(page, pause_ms, prefer_last=False)

    show_banner(page, "Review · diversity impact")
    heading = page.get_by_text("Diversity Impact & Formula")
    if heading.count():
        try:
            heading.first.scroll_into_view_if_needed(timeout=5000)
            page.wait_for_timeout(pause_ms)
        except Exception:
            pass
        expand_expander(page, "Show Diversity Impact Table")
        page.wait_for_timeout(pause_ms // 2)
        scroll_main(page, 300)
        page.wait_for_timeout(pause_ms // 3)

    review_taste_map(page, pause_ms)


def review_nlp_outputs(page, pause_ms: int) -> None:
    scroll_to_top(page)
    show_banner(page, "Review · parsed intent")
    intent = page.get_by_text("Parsed intent")
    if intent.count():
        try:
            intent.first.scroll_into_view_if_needed(timeout=5000)
            page.wait_for_timeout(pause_ms)
        except Exception:
            pass

    show_banner(page, "Review · NLP recommendations")
    rec_label = page.get_by_text("Recommendations from parsed intent")
    if rec_label.count():
        try:
            rec_label.first.scroll_into_view_if_needed(timeout=5000)
            page.wait_for_timeout(pause_ms // 2)
        except Exception:
            pass
    _scroll_visible_dataframe(page, pause_ms, prefer_last=True)
    review_taste_map(page, pause_ms)


def review_agent_outputs(page, pause_ms: int) -> None:
    scroll_to_top(page)
    show_banner(page, "Review · agent trace")
    trace = page.get_by_text("Agent trace — thinking, tools & observations")
    if trace.count():
        try:
            expand_expander(page, "Agent trace — thinking, tools & observations")
            page.wait_for_timeout(pause_ms)
            scroll_gradually(page, steps=2, step_px=300, pause_ms=pause_ms // 2)
        except Exception:
            pass

    show_banner(page, "Review · agent reply")
    reply = page.locator("h4").filter(has_text=re.compile(r"^Agent reply$"))
    if reply.count():
        try:
            reply.first.scroll_into_view_if_needed(timeout=5000)
            page.wait_for_timeout(pause_ms)
        except Exception:
            pass

    show_banner(page, "Review · agent recommendations")
    _scroll_visible_dataframe(page, pause_ms, prefer_last=True)
    review_taste_map(page, pause_ms)


def setup_agent_mode(page, user_id: int) -> None:
    click_recommendation_mode(page, "Agent (NLP)")
    select_labeled_option(page, "NLP Runtime", "Local LLM")
    set_user_id(page, user_id)


def set_agent_query(page, text: str) -> None:
    field = page.get_by_placeholder("e.g., top 5 action movies for user 120 with tuned model")
    field.click()
    field.fill(text)
    wait_streamlit(page, 400)


def toggle_labeled_checkbox(page, label: str, wanted: bool) -> None:
    scroll_to_top(page)
    wait_streamlit(page, 400)
    cb = page.get_by_role("checkbox", name=label)
    if not cb.count():
        return
    try:
        if cb.is_checked() != wanted:
            page.locator("label").filter(has_text=label).first.click()
            wait_streamlit(page)
    except Exception:
        if wanted:
            cb.check(force=True)
        else:
            cb.uncheck(force=True)
        wait_streamlit(page)


def run_agent_nlp_parse(page, pause_ms: int, user_id: int) -> None:
    show_banner(page, "Agent · Parse Query (NLP) · Local LLM")
    setup_agent_mode(page, user_id)
    toggle_labeled_checkbox(page, "Multi-step tool agent (Ollama)", wanted=False)

    set_agent_query(
        page,
        f"top 5 action movies for user {user_id} using gradient boosting with diverse recommendations",
    )
    page.get_by_role("button", name="Parse Query (NLP)").click()
    page.wait_for_selector("text=Parsed intent", timeout=180000)
    wait_streamlit(page, pause_ms // 2)
    review_nlp_outputs(page, pause_ms)


def wait_agent_run_complete(page, timeout_ms: int = 300000) -> None:
    """Wait for tool agent SSE/one-shot run to finish (avoid hidden 'Agent reply' in help text)."""
    done = page.get_by_text("Agent status: done")
    failed = page.get_by_text("Agent status: failed")
    reply_heading = page.locator("h4").filter(has_text=re.compile(r"^Agent reply$"))
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        if done.count() and done.first.is_visible():
            return
        if failed.count() and failed.first.is_visible():
            return
        if reply_heading.count() and reply_heading.first.is_visible():
            return
        page.wait_for_timeout(800)
    raise TimeoutError("Tool agent did not complete within timeout")


def run_agent_tool_multi_step(page, pause_ms: int, user_id: int) -> None:
    show_banner(page, "Agent · Multi-step tool agent (Ollama)")
    scroll_to_top(page)
    setup_agent_mode(page, user_id)
    toggle_labeled_checkbox(page, "Multi-step tool agent (Ollama)", wanted=True)
    toggle_labeled_checkbox(page, "Stop after first recommendations", wanted=True)

    set_agent_query(
        page,
        f"Recommend 5 diverse horror movies for user {user_id} unlike my usual training taste",
    )
    page.get_by_role("button", name="Run tool agent").click()
    wait_agent_run_complete(page, timeout_ms=300000)
    wait_streamlit(page, pause_ms // 2)
    review_agent_outputs(page, pause_ms)


def run_agent_section(page, pause_ms: int, user_id: int) -> None:
    run_agent_nlp_parse(page, pause_ms, user_id)
    run_agent_tool_multi_step(page, pause_ms, user_id)


def browse_tab_with_scroll(page, tab_name: str, banner: str, pause_ms: int) -> None:
    show_banner(page, banner)
    click_tab(page, tab_name)
    scroll_to_top(page)
    page.wait_for_timeout(pause_ms // 2)
    if tab_name == "Ollama":
        refresh = page.get_by_role("button", name="Refresh Ollama snapshot")
        if refresh.count():
            refresh.click()
            wait_streamlit(page, 1200)
    scroll_gradually(page, steps=6 if tab_name in {"Model Visualizers", "Lifecycle Evidence"} else 4)
    scroll_to_top(page)
    page.wait_for_timeout(pause_ms // 2)


def click_recommendation_mode(page, mode: str) -> None:
    page.locator('[data-testid="stRadio"]').get_by_text(mode, exact=True).click()
    wait_streamlit(page)


def run_manual_model_comparison(
    page, model_labels: list[str], pause_ms: int, user_id: int, diversity: float
) -> None:
    click_recommendation_mode(page, "Manual")
    set_user_id(page, user_id)
    set_diversity(page, diversity)

    for idx, label in enumerate(model_labels, start=1):
        show_banner(page, f"Manual · {label.split('(')[0].strip()} ({idx}/{len(model_labels)})")
        select_labeled_option(page, "Model", label)
        page.get_by_role("button", name="Get Recommendations").click()
        page.wait_for_selector("text=Returned", timeout=60000)
        wait_streamlit(page, pause_ms // 2)
        review_recommendation_outputs(page, pause_ms)


def browse_other_tabs(page, pause_ms: int) -> None:
    tab_plan = [
        ("Model Inspector", "Model Inspector · metrics & artifacts"),
        ("Embedding Space", "Embedding Space · user taste clusters"),
        ("Model Visualizers", "Model Visualizers · NCF & GB charts"),
        ("Lifecycle Evidence", "Lifecycle Evidence · phase artifacts"),
        ("AI Concepts", "AI Concepts · RMSE / Precision@K"),
        ("Ollama", "Ollama monitor · local LLM health"),
        ("System", "System · API health"),
    ]
    for tab_name, banner in tab_plan:
        browse_tab_with_scroll(page, tab_name, banner, pause_ms)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record MovieMind demo video with Playwright")
    parser.add_argument("--user-id", type=int, default=1161)
    parser.add_argument("--diversity", type=float, default=0.35)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pause-ms", type=int, default=3500, help="Pause after each major step")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument(
        "--manage-services",
        action="store_true",
        help="Start/stop API + Streamlit from this script (recommended)",
    )
    parser.add_argument(
        "--with-ollama",
        action="store_true",
        help="With --manage-services, run setup_local_ollama.sh first",
    )
    return parser.parse_args()


def main() -> int:
    cli = parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Install Playwright: pip install playwright && playwright install chromium", file=sys.stderr)
        return 1

    services: Optional[ServiceManager] = None
    try:
        if cli.manage_services:
            services = ServiceManager(with_ollama=cli.with_ollama)
            print("Starting API + Streamlit for demo capture...")
            services.start()
        else:
            wait_for_api()
            wait_for_ui(30)

        model_labels = fetch_model_labels()
        print(f"Available models ({len(model_labels)}): {', '.join(model_labels)}")

        cli.output.parent.mkdir(parents=True, exist_ok=True)
        tmp_video_dir = cli.output.parent / "_playwright_videos"
        tmp_video_dir.mkdir(parents=True, exist_ok=True)

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, slow_mo=120)
            context = browser.new_context(
                viewport={"width": cli.width, "height": cli.height},
                record_video_dir=str(tmp_video_dir),
                record_video_size={"width": cli.width, "height": cli.height},
            )
            page = context.new_page()
            page.set_default_timeout(120000)

            show_banner(page, "MovieMind · launching")
            page.goto(UI_URL, wait_until="domcontentloaded")
            wait_streamlit(page, 2500)

            show_banner(page, "Recommend · Manual mode · all models")
            run_manual_model_comparison(page, model_labels, cli.pause_ms, cli.user_id, cli.diversity)

            run_agent_section(page, cli.pause_ms, cli.user_id)

            browse_other_tabs(page, max(2500, cli.pause_ms - 500))

            show_banner(page, "Demo complete")
            wait_streamlit(page, 2000)

            video_path = page.video.path()
            context.close()
            browser.close()

        if not video_path or not Path(video_path).exists():
            print("Playwright did not produce a video file.", file=sys.stderr)
            return 1

        shutil.move(str(video_path), cli.output)
        for leftover in tmp_video_dir.glob("*"):
            leftover.unlink(missing_ok=True)
        try:
            tmp_video_dir.rmdir()
        except OSError:
            pass

        size_kb = cli.output.stat().st_size / 1024
        print(f"Wrote {cli.output} ({size_kb:.0f} KiB)")
        return 0
    finally:
        if services is not None:
            print("Stopping demo services...")
            services.stop()


if __name__ == "__main__":
    raise SystemExit(main())
