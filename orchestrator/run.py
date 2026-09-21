"""Entry point: run one cell of the experiment grid.

    python -m orchestrator.run --config configs/sonnet_3.yaml
    python -m orchestrator.run --config configs/smoke.yaml --fake-gpu
    python -m orchestrator.run --resume-run runs/sonnet_3-20260921-1400
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from .arena import Arena
from .config import CellConfig, load_cell

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run one multi-agent autoresearch cell.")
    ap.add_argument("--config", type=Path, help="cell config YAML")
    ap.add_argument("--resume-run", type=Path, help="resume an interrupted run directory")
    ap.add_argument("--run-dir", type=Path, help="override the run directory")
    ap.add_argument("--fake-gpu", action="store_true",
                    help="simulate training; exercises the whole loop without an H100")
    ap.add_argument("--fake-agents", action="store_true",
                    help="canned agent output instead of API calls; free, and with "
                         "--fake-gpu gives an end-to-end test that costs nothing")
    ap.add_argument("--repo", type=Path, help="override autoresearch_repo")
    ap.add_argument("--budget", type=int, help="override train_run_budget (for smoke tests)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the rendered prompts, then exit")
    return ap.parse_args()


def load(args: argparse.Namespace) -> tuple[CellConfig, Path, bool]:
    if args.resume_run:
        run_dir = args.resume_run
        cfg_dict = json.loads((run_dir / "provenance.json").read_text())["cell"]
        agents = cfg_dict.pop("agents")
        from .config import AgentSpec

        cfg = CellConfig(agents=[AgentSpec(**a) for a in agents], **cfg_dict)
        return cfg, run_dir, True

    if not args.config:
        raise SystemExit("one of --config or --resume-run is required")

    cfg = load_cell(args.config)
    if args.fake_gpu:
        cfg.fake_gpu = True
    if args.fake_agents:
        # Swap the harness, keep the model id: provenance still records which
        # cell this was pretending to be.
        cfg.agents = [replace(a, harness="fake") for a in cfg.agents]
    if args.repo:
        cfg.autoresearch_repo = str(args.repo)
    if args.budget:
        cfg.train_run_budget = args.budget

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.run_dir or REPO_ROOT / "runs" / f"{cfg.cell_id}-{stamp}"
    return cfg, Path(run_dir), False


def main() -> int:
    args = parse_args()
    cfg, run_dir, resume = load(args)

    if args.dry_run:
        print(cfg.summary())
        print(json.dumps(cfg.to_dict(), indent=2))
        return 0

    print(f"run directory: {run_dir}")
    arena = Arena(cfg, run_dir, resume=resume)
    state = arena.run()
    return 0 if state.stop_reason else 1


if __name__ == "__main__":
    raise SystemExit(main())
