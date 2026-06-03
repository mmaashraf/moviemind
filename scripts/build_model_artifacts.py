#!/usr/bin/env python3
r"""
MovieMind: build local model artifacts under models/ (gitignored binaries).

Run from the moviemind/ directory:

  python3 scripts/build_model_artifacts.py <command>

------------------------------------------------------------------------------
Commands (each wraps one script under src/, except ``all``)
------------------------------------------------------------------------------

  features   Phase 3 — builds ``data/processed/train_features.csv`` and test CSVs.

  ml         Phase 4 — sklearn models → ``models/*.pkl`` (incl. gradient_boosting).

  dl         Phase 5 — baseline Neural Collaborative Filtering → ``models/ncf_model.pt``
             (fixed architecture from dl_model.py).

  tune-dl    Phase 5b — Optuna searches hyperparameters (many trials). Each trial
             trains a candidate NCF for only a *few* epochs to score that setting
             quickly. Writes ``models/best_dl_params.txt`` (winner config).
             Does *not* write ``ncf_tuned_best.pt``.

  post       Phase 6 — reads ``best_dl_params.txt``, builds that architecture, then
             runs *one* training job with those fixed parameters for another small
             fixed number of epochs, and saves ``models/ncf_tuned_best.pt`` plus
             embedding/XAI outputs under ``evidence/phase6/``. That step is what we
             mean by “retrain the tuned NCF for a few epochs”: not more Optuna, just
             fitting one final checkpoint using the winning hyperparameters from tuning.

  all        Runs in order: features → ml → dl → tune-dl → post.
             Use ``--skip-tune-dl`` (or env ``MOVIEMIND_SKIP_TUNE_DL=1``) to skip
             Optuna if ``models/best_dl_params.txt`` already exists.

------------------------------------------------------------------------------
Optuna vs “best model”
------------------------------------------------------------------------------

  ``tune-dl`` uses Optuna to find *hyperparameters* that minimize validation RMSE
  among the trials you run (search space + trial count). Each trial is a *cheap*
  surrogate (few epochs), so the winner is the best *configuration in that search*,
  not proof of global optimum.

  The checkpoint you deploy for “tuned NCF” is produced by ``post``, which trains
  once with those params and saves ``ncf_tuned_best.pt``. Quality depends on search
  settings, epochs in ``post``, and data — not only Optuna.

------------------------------------------------------------------------------
Git / binaries
------------------------------------------------------------------------------

  Generally do not commit ``*.pkl`` / ``*.pt`` — large, poor diffs, pickle/version
  sensitivity. Generate locally or ship via releases/object storage.

------------------------------------------------------------------------------
Examples
------------------------------------------------------------------------------

  python3 scripts/build_model_artifacts.py all
  python3 scripts/build_model_artifacts.py all --skip-tune-dl
  MOVIEMIND_SKIP_TUNE_DL=1 python3 scripts/build_model_artifacts.py all
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


class PhaseFailed(Exception):
    """Training subprocess exited non-zero."""

    def __init__(self, code: int) -> None:
        self.code = code
        super().__init__(f"subprocess exited with status {code}")


def _run_phase(label: str, script: str) -> None:
    script_path = ROOT / script
    print(f"[MovieMind] {label}")
    proc = subprocess.run([sys.executable, str(script_path)], cwd=ROOT)
    if proc.returncode != 0:
        print(
            f"\n[MovieMind] Phase failed: {script} (exit {proc.returncode}). "
            "Fix the error above, then re-run the same command or run a single phase, e.g.:\n"
            f"  python3 scripts/build_model_artifacts.py ml\n",
            file=sys.stderr,
        )
        raise PhaseFailed(proc.returncode) from None


def cmd_features(_: argparse.Namespace) -> None:
    _run_phase("Phase 3: features", "src/features.py")


def cmd_ml(_: argparse.Namespace) -> None:
    _run_phase(
        "Phase 4: ML models (writes *.pkl including gradient_boosting.pkl)",
        "src/ml_models.py",
    )


def cmd_dl(_: argparse.Namespace) -> None:
    _run_phase("Phase 5: baseline NCF (ncf_model.pt)", "src/dl_model.py")


def cmd_tune_dl(_: argparse.Namespace) -> None:
    _run_phase("Phase 5b: Optuna tune_dl (long run)", "src/tune_dl.py")


def cmd_post(_: argparse.Namespace) -> None:
    _run_phase(
        "Phase 6: post_analysis (ncf_tuned_best.pt + XAI evidence)",
        "src/post_analysis.py",
    )


def cmd_all(args: argparse.Namespace) -> None:
    skip = args.skip_tune_dl or os.environ.get("MOVIEMIND_SKIP_TUNE_DL", "").strip() == "1"
    phases = [
        ("features", cmd_features),
        ("ml", cmd_ml),
        ("dl", cmd_dl),
    ]
    for name, fn in phases:
        try:
            fn(args)
        except PhaseFailed:
            print(f"[MovieMind] Stopped after failed phase: {name}", file=sys.stderr)
            raise

    params_file = ROOT / "models" / "best_dl_params.txt"
    if skip:
        if not params_file.is_file():
            print(
                "[MovieMind] ERROR: --skip-tune-dl / MOVIEMIND_SKIP_TUNE_DL=1 "
                "but models/best_dl_params.txt is missing.\n"
                "  Run: python3 scripts/build_model_artifacts.py tune-dl",
                file=sys.stderr,
            )
            raise SystemExit(1)
        print("[MovieMind] Skipping tune_dl; using existing models/best_dl_params.txt")
    else:
        try:
            cmd_tune_dl(args)
        except PhaseFailed:
            print("[MovieMind] Stopped after failed phase: tune-dl", file=sys.stderr)
            raise

    try:
        cmd_post(args)
    except PhaseFailed:
        print("[MovieMind] Stopped after failed phase: post", file=sys.stderr)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Chain MovieMind training scripts to fill models/ (gitignored). "
            "Full command reference is in the module docstring below."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_all = sub.add_parser(
        "all",
        help="Run features → ml → dl → tune-dl → post (full pipeline).",
        description=(
            "Equivalent to running phases 3–6 in order. Optuna (tune-dl) is slow; "
            "use --skip-tune-dl if models/best_dl_params.txt already exists."
        ),
    )
    p_all.add_argument(
        "--skip-tune-dl",
        action="store_true",
        help="Reuse existing models/best_dl_params.txt; skip Optuna (also: MOVIEMIND_SKIP_TUNE_DL=1).",
    )
    p_all.set_defaults(func=cmd_all)

    sub.add_parser(
        "features",
        help="Phase 3: processed CSVs (src/features.py).",
        description="Build data/processed train/val/test feature tables.",
    ).set_defaults(func=cmd_features)
    sub.add_parser(
        "ml",
        help="Phase 4: sklearn *.pkl models (src/ml_models.py).",
        description="Train baseline + linear + RF + gradient boosting; writes models/*.pkl.",
    ).set_defaults(func=cmd_ml)
    sub.add_parser(
        "dl",
        help="Phase 5: baseline NCF checkpoint (src/dl_model.py).",
        description="Train fixed-architecture NCF; writes models/ncf_model.pt.",
    ).set_defaults(func=cmd_dl)
    sub.add_parser(
        "tune-dl",
        help="Phase 5b: Optuna search → best_dl_params.txt (src/tune_dl.py).",
        description=(
            "Many short trials to pick hyperparameters; writes models/best_dl_params.txt. "
            "Does not produce ncf_tuned_best.pt (see post)."
        ),
    ).set_defaults(func=cmd_tune_dl)
    sub.add_parser(
        "post",
        help="Phase 6: tuned NCF weights + XAI (src/post_analysis.py).",
        description=(
            "Load winning params from best_dl_params.txt, train one final NCF, "
            "save models/ncf_tuned_best.pt and evidence/phase6/ artifacts."
        ),
    ).set_defaults(func=cmd_post)

    args = parser.parse_args()
    try:
        args.func(args)
    except PhaseFailed as exc:
        raise SystemExit(exc.code) from None
    print(f"[MovieMind] Done: {args.command}")


if __name__ == "__main__":
    main()
