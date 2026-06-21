#!/usr/bin/env python3
"""
Record MovieMind capstone notebook execution in JupyterLab (Playwright → WebM).

Prerequisites:
  source .venv/bin/activate
  pip install -r requirements-demo.txt && playwright install chromium
  data/ + models/ artifacts present (download_review_artifacts.sh)

Usage:
  python scripts/record_notebook_video.py --manage-services
  python scripts/record_notebook_video.py --manage-services --pause-ms 2500
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "MovieMind_capstone.ipynb"
DEFAULT_OUT = ROOT / "evidence" / "demo" / "video" / "moviemind_capstone_notebook.webm"
LAB_PORT = 8889
LAB_URL = f"http://127.0.0.1:{LAB_PORT}/lab"
NOTEBOOK_TREE_URL = f"{LAB_URL}/tree/notebooks/MovieMind_capstone.ipynb"


def show_banner(page, text: str) -> None:
    page.evaluate(
        """(text) => {
            let el = document.getElementById('moviemind-demo-banner');
            if (!el) {
                el = document.createElement('div');
                el.id = 'moviemind-demo-banner';
                el.style.cssText = [
                    'position:fixed','top:14px','left:50%','transform:translateX(-50%)',
                    'z-index:999999','background:#1a3d2e','color:#fff','padding:10px 22px',
                    'border-radius:8px','font:600 17px -apple-system,BlinkMacSystemFont,sans-serif',
                    'box-shadow:0 4px 18px rgba(0,0,0,.35)','pointer-events:none'
                ].join(';');
                document.body.appendChild(el);
            }
            el.textContent = text;
        }""",
        text,
    )


def wait_for_lab(max_sec: int = 90) -> None:
    deadline = time.time() + max_sec
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(LAB_URL, timeout=4) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(2)
    raise RuntimeError(f"JupyterLab not ready at {LAB_URL}")


class LabManager:
    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None

    def start(self) -> None:
        env = {
            **os.environ,
            "MOVIEMIND_SKIP_TUNE_DL": os.environ.get("MOVIEMIND_SKIP_TUNE_DL", "1"),
            "MOVIEMIND_SKIP_TUNE_ML": os.environ.get("MOVIEMIND_SKIP_TUNE_ML", "1"),
            "MOVIEMIND_SKIP_POST": os.environ.get("MOVIEMIND_SKIP_POST", "1"),
        }
        cmd = [
            sys.executable,
            "-m",
            "jupyter",
            "lab",
            f"--port={LAB_PORT}",
            "--no-browser",
            f"--ServerApp.root_dir={ROOT}",
            "--IdentityProvider.token=",
            "--ServerApp.password=",
            "--LabApp.open_browser=False",
        ]
        log_dir = ROOT / "evidence" / "runtime"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_f = open(log_dir / "demo_jupyter.log", "w", encoding="utf-8")  # noqa: SIM115
        self._proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        wait_for_lab(120)

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.send_signal(signal.SIGTERM)
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None


def dismiss_jupyter_dialogs(page) -> None:
    for label in ("Trust", "Yes", "OK", "Dismiss"):
        btn = page.get_by_role("button", name=re.compile(label, re.I))
        if btn.count():
            try:
                btn.first.click(timeout=2000)
                page.wait_for_timeout(500)
            except Exception:
                pass


def wait_cell_idle(page, cell_locator, timeout_ms: int = 600000) -> None:
    """Wait until a JupyterLab code cell is not running ([*] prompt gone)."""
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        try:
            prompt = cell_locator.locator(".jp-InputPrompt").first
            text = (prompt.inner_text(timeout=1000) or "").strip()
            if text and "[*]" not in text:
                return
            cls = cell_locator.get_attribute("class") or ""
            if "jp-mod-running" not in cls and text and "[*]" not in text:
                return
        except Exception:
            pass
        page.wait_for_timeout(400)
    raise TimeoutError("Cell execution timed out")


def scroll_cell_into_view(page, cell_locator) -> None:
    cell_locator.scroll_into_view_if_needed()
    page.wait_for_timeout(350)
    page.evaluate("window.scrollBy(0, -80)")  # leave room for banner
    page.wait_for_timeout(200)


def review_cell_outputs(page, cell_locator, pause_ms: int) -> None:
    scroll_cell_into_view(page, cell_locator)
    outputs = cell_locator.locator(".jp-OutputArea")
    if outputs.count() == 0:
        page.wait_for_timeout(pause_ms // 2)
        return
    outputs.first.scroll_into_view_if_needed()
    page.wait_for_timeout(pause_ms)
    for _ in range(3):
        page.evaluate("window.scrollBy(0, 320)")
        page.wait_for_timeout(pause_ms // 3)


def open_notebook_in_lab(page) -> None:
    page.goto(NOTEBOOK_TREE_URL, wait_until="domcontentloaded")
    page.wait_for_timeout(6000)
    dismiss_jupyter_dialogs(page)

    if page.locator(".jp-CodeCell").count() == 0:
        link = page.get_by_role("link", name=re.compile(r"MovieMind_capstone\.ipynb"))
        if link.count():
            link.first.dblclick()
            page.wait_for_timeout(8000)
        else:
            page.locator('[data-file-type="notebook"]').filter(
                has_text="MovieMind_capstone"
            ).first.dblclick()
            page.wait_for_timeout(8000)

    dismiss_jupyter_dialogs(page)
    page.wait_for_selector(".jp-CodeCell", timeout=120000)


def run_notebook_cells(page, pause_ms: int, cell_timeout_ms: int) -> None:
    open_notebook_in_lab(page)
    code_cells = page.locator(".jp-CodeCell").all()
    total = len(code_cells)
    print(f"Running {total} code cells...")

    for idx, cell in enumerate(code_cells, start=1):
        show_banner(page, f"Cell {idx}/{total} · executing")
        scroll_cell_into_view(page, cell)
        cell.click()
        page.wait_for_timeout(300)
        page.keyboard.press("Shift+Enter")
        wait_cell_idle(page, cell, timeout_ms=cell_timeout_ms)
        show_banner(page, f"Cell {idx}/{total} · output")
        review_cell_outputs(page, cell, pause_ms)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record capstone notebook execution video")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--pause-ms", type=int, default=2200, help="Pause on each cell output")
    p.add_argument("--cell-timeout-ms", type=int, default=600000, help="Max wait per cell")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=900)
    p.add_argument(
        "--manage-services",
        action="store_true",
        help="Start/stop JupyterLab from this script (recommended)",
    )
    return p.parse_args()


def main() -> int:
    cli = parse_args()

    if not NOTEBOOK.exists():
        print(f"Missing notebook: {NOTEBOOK}", file=sys.stderr)
        return 1

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Install: pip install playwright && playwright install chromium", file=sys.stderr)
        return 1

    lab: Optional[LabManager] = None
    try:
        if cli.manage_services:
            print("Starting JupyterLab for notebook capture...")
            lab = LabManager()
            lab.start()
        else:
            wait_for_lab(30)

        cli.output.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = cli.output.parent / "_playwright_notebook_videos"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, slow_mo=80)
            context = browser.new_context(
                viewport={"width": cli.width, "height": cli.height},
                record_video_dir=str(tmp_dir),
                record_video_size={"width": cli.width, "height": cli.height},
            )
            page = context.new_page()
            page.set_default_timeout(120000)

            show_banner(page, "MovieMind Capstone Notebook")
            run_notebook_cells(page, cli.pause_ms, cli.cell_timeout_ms)

            show_banner(page, "Notebook run complete")
            page.wait_for_timeout(2500)

            video_path = page.video.path()
            context.close()
            browser.close()

        if not video_path or not Path(video_path).exists():
            print("No video produced.", file=sys.stderr)
            return 1

        shutil.move(str(video_path), cli.output)
        for f in tmp_dir.glob("*"):
            f.unlink(missing_ok=True)
        try:
            tmp_dir.rmdir()
        except OSError:
            pass

        print(f"Wrote {cli.output} ({cli.output.stat().st_size / 1024:.0f} KiB)")
        return 0
    finally:
        if lab is not None:
            print("Stopping JupyterLab...")
            lab.stop()


if __name__ == "__main__":
    raise SystemExit(main())
