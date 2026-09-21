"""The round loop.

Agents propose in parallel, respond in parallel, then implement and train in
parallel -- with ``arena-train`` serializing them onto the one GPU.  The best
candidate of the round advances the shared lineage, which is what makes a cell
a single comparable research trajectory rather than N private hill-climbs.
"""

from __future__ import annotations

import json
import shutil
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from . import phases
from .config import AgentSpec, CellConfig
from .gpu import assert_no_overlap, parse_run_log, read_arena_run
from .harness import HarnessError, build_harness
from .sharedlog import SharedLog
from .state import RunState
from .worktrees import WorktreeManager, git


@dataclass
class Candidate:
    agent: str
    variant: int
    worktree: Path
    commit: str = ""
    val_bpb: float | None = None
    peak_vram_mb: float | None = None
    status: str = "crash"
    title: str = ""
    description: str = ""
    reflection: str = ""
    error: str = ""
    attempts: int = 0
    cost_usd: float = 0.0
    num_turns: int = 0

    @property
    def improved_on(self):
        def _cmp(baseline: float) -> bool:
            return self.status == "ok" and self.val_bpb is not None and self.val_bpb < baseline
        return _cmp


class Arena:
    def __init__(self, cfg: CellConfig, run_dir: Path, resume: bool = False):
        self.cfg = cfg
        # Absolute: git resolves worktree paths against the repo, not our cwd.
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "rounds").mkdir(exist_ok=True)
        (self.run_dir / "transcripts").mkdir(exist_ok=True)

        self.log = SharedLog(self.run_dir)
        self.tag = self.run_dir.name
        self.trees = WorktreeManager(cfg.repo_path, self.run_dir, self.tag)
        # Distinct seeds so fake agents do not all propose the same idea.
        self.harnesses = {
            a.id: build_harness(a.harness, seed=cfg.seed + i)
            for i, a in enumerate(cfg.agents)
        }
        self.settings_path = phases.write_agent_settings(self.run_dir)
        self.state = RunState.load(self.run_dir) if resume else self._fresh_state()
        if not resume:
            self._init_budget()
            self._record_provenance()

    # -- setup -----------------------------------------------------------

    def _baseline(self) -> tuple[str, float]:
        """The starting commit and its val_bpb.

        Measured once per GPU box by ``scripts/setup_gpu_box.sh`` and shared by
        every cell, so all cells begin from an identical number.
        """
        pinned = self.cfg.autoresearch_commit or "HEAD"
        commit = self.trees.resolve(pinned)

        for path in (self.cfg.repo_path / "baseline.json", self.run_dir / "baseline.json"):
            if path.exists():
                data = json.loads(path.read_text())
                if data.get("commit", commit)[:7] != commit[:7]:
                    raise SystemExit(
                        f"{path} was measured at commit {data.get('commit', '')[:7]} but the "
                        f"cell is pinned to {commit[:7]}. Re-run scripts/setup_gpu_box.sh."
                    )
                return commit, float(data["val_bpb"])

        if self.cfg.fake_gpu:
            return commit, 0.997900

        raise SystemExit(
            "No baseline.json found. Run scripts/setup_gpu_box.sh first so every "
            "cell starts from the same measured baseline."
        )

    def _fresh_state(self) -> RunState:
        commit, bpb = self._baseline()
        state = RunState(
            cell_id=self.cfg.cell_id,
            tag=self.tag,
            origin_commit=commit,
            baseline_commit=commit,
            baseline_bpb=bpb,
            started_at=time.time(),
        )
        state.save(self.run_dir)
        return state

    def _init_budget(self) -> None:
        (self.run_dir / "budget.json").write_text(
            json.dumps(
                {"budget_s": self.cfg.gpu_budget_s, "spent_s": 0.0, "runs": 0},
                indent=2,
            )
        )

    def _record_provenance(self) -> None:
        import subprocess

        claude_version = subprocess.run(
            ["claude", "--version"], capture_output=True, text=True
        ).stdout.strip()
        (self.run_dir / "provenance.json").write_text(
            json.dumps(
                {
                    "cell": self.cfg.to_dict(),
                    "config_fingerprint": self.cfg.fingerprint(),
                    "claude_version": claude_version,
                    "autoresearch_commit": self.state.origin_commit,
                    "started_at": self.state.started_at,
                },
                indent=2,
            )
        )

    # -- budget ----------------------------------------------------------

    def budget(self) -> dict:
        return json.loads((self.run_dir / "budget.json").read_text())

    def gpu_exhausted(self) -> bool:
        b = self.budget()
        return b["spent_s"] >= b["budget_s"]

    # -- context ---------------------------------------------------------

    def _context(self, round_idx: int) -> phases.PhaseContext:
        train_py = git(self.cfg.repo_path, "show", f"{self.state.baseline_commit}:train.py")
        diff = self.trees.diff(self.state.origin_commit, self.state.baseline_commit)
        ctx = phases.PhaseContext(
            cfg=self.cfg,
            round_idx=round_idx,
            log=self.log,
            run_dir=self.run_dir,
            train_py=train_py,
            diff_vs_origin=diff,
            settings_path=self.settings_path,
            baseline_bpb=self.state.baseline_bpb,
        )
        return ctx

    def _charge(self, cost: float) -> None:
        self.state.total_cost_usd += cost

    def _parallel(self, fn, items, label: str) -> dict:
        """Run one call per agent, tolerating individual failures."""
        results: dict = {}
        with ThreadPoolExecutor(max_workers=max(1, len(items))) as pool:
            futures = {pool.submit(fn, item): item for item in items}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    results[key] = future.result()
                except Exception as exc:  # one agent failing must not kill the cell
                    print(f"  ! {label} failed for {key}: {exc}")
                    (self.run_dir / "errors.log").open("a").write(
                        f"\n=== {label} {key} ===\n{traceback.format_exc()}\n"
                    )
                    results[key] = None
        return results

    # -- the loop ---------------------------------------------------------

    def run(self) -> RunState:
        print(self.cfg.summary())
        while True:
            r = self.state.round_idx
            if r >= self.cfg.max_rounds:
                self.state.stop_reason = "round cap reached"
                break
            if self.gpu_exhausted():
                self.state.stop_reason = "GPU budget exhausted"
                break
            if self.state.total_cost_usd >= self.cfg.cell_budget_usd:
                self.state.stop_reason = "agent cost ceiling reached"
                break

            self.run_round(r)
            self.state.round_idx = r + 1
            self.state.completed_rounds.append(r)
            self.state.save(self.run_dir)

        self.state.finished_at = time.time()
        self.state.save(self.run_dir)
        self._finish()
        return self.state

    def run_round(self, r: int) -> None:
        b = self.budget()
        print(
            f"\n=== round {r} | baseline val_bpb {self.state.baseline_bpb:.6f} "
            f"| {(b['budget_s'] - b['spent_s']) / 60:.0f} GPU-min left "
            f"| ${self.state.total_cost_usd:.2f} spent ==="
        )
        self.log.append(
            "round_start",
            round=r,
            baseline_commit=self.state.baseline_commit,
            baseline_bpb=self.state.baseline_bpb,
            gpu_remaining_s=b["budget_s"] - b["spent_s"],
        )

        ctx = self._context(r)
        self.log.publish(upto_round=r)

        # 1. propose ------------------------------------------------------
        print("  propose ...")
        proposals = self._parallel(
            lambda a: phases.run_propose(self.harnesses[a.id], a, ctx),
            self.cfg.agents,
            "propose",
        )
        sessions: dict[str, str] = {}
        titles: dict[str, str] = {}
        for agent, result in proposals.items():
            if result is None:
                continue
            self._charge(result.cost_usd)
            sessions[agent.id] = result.session_id
            data = result.structured or {}
            titles[agent.id] = data.get("title", "(unparsed)")
            self.log.append(
                "proposal",
                round=r,
                agent=agent.id,
                model=agent.model,
                title=data.get("title", ""),
                justification=data.get("justification", ""),
                detail=data.get("detail", ""),
                notes=data.get("notes", ""),
                cost_usd=result.cost_usd,
                session_id=result.session_id,
                models_used=result.models_used,
            )
            print(f"    {agent.id}: {titles[agent.id]}")

        # 2. respond ------------------------------------------------------
        if self.cfg.respond_enabled:
            print("  respond ...")
            self.log.publish(upto_round=r, include_proposals_for=r)
            responses = self._parallel(
                lambda a: phases.run_respond(
                    self.harnesses[a.id], a, ctx, sessions.get(a.id, ""), titles.get(a.id, "")
                ),
                [a for a in self.cfg.agents if a.id in sessions],
                "respond",
            )
            for agent, result in responses.items():
                if result is None:
                    continue
                self._charge(result.cost_usd)
                sessions[agent.id] = result.session_id or sessions[agent.id]
                data = result.structured or {}
                self.log.append(
                    "response",
                    round=r,
                    agent=agent.id,
                    text=data.get("response", ""),
                    notes=data.get("notes", ""),
                    cost_usd=result.cost_usd,
                )
        else:
            self.log.append("response_skipped", round=r, reason="single-agent cell")

        # 3. finalize + run ------------------------------------------------
        print(f"  finalize + train ({self.cfg.runs_per_round} candidates, GPU serialized) ...")
        self.log.publish(upto_round=r, include_proposals_for=r)
        slots = [
            (agent, variant)
            for agent in self.cfg.agents
            if agent.id in sessions
            for variant in range(self.cfg.bon)
        ]
        finals = self._parallel(
            lambda slot: self._execute(ctx, slot[0], slot[1], sessions[slot[0].id]),
            slots,
            "finalize",
        )

        candidates = [c for c in finals.values() if c is not None]
        for c in candidates:
            self._charge(c.cost_usd)
            self.log.append(
                "candidate",
                round=r,
                agent=c.agent,
                variant=c.variant,
                commit=c.commit,
                val_bpb=c.val_bpb,
                peak_vram_mb=c.peak_vram_mb,
                status=c.status,
                title=c.title,
                description=c.description,
                reflection=c.reflection,
                error=c.error[:2000],
                attempts=c.attempts,
                cost_usd=c.cost_usd,
                num_turns=c.num_turns,
            )
            metric = f"{c.val_bpb:.6f}" if c.val_bpb is not None else "crash"
            print(f"    {c.agent}/v{c.variant}: {metric}  {c.description[:60]}")

        # 4. select --------------------------------------------------------
        winners = [
            c for c in candidates
            if c.status == "ok" and c.val_bpb is not None and c.val_bpb < self.state.baseline_bpb
        ]
        if winners:
            best = min(winners, key=lambda c: c.val_bpb)
            self.state.baseline_commit = best.commit
            self.state.baseline_bpb = best.val_bpb
            self.log.append(
                "round_end",
                round=r,
                improved=True,
                winner={"agent": best.agent, "variant": best.variant, "commit": best.commit},
                new_baseline_bpb=best.val_bpb,
                n_candidates=len(candidates),
                n_crashes=sum(1 for c in candidates if c.status == "crash"),
            )
            print(f"  -> {best.agent}/v{best.variant} advances the lineage "
                  f"({best.val_bpb:.6f})")
        else:
            self.log.append(
                "round_end",
                round=r,
                improved=False,
                winner=None,
                new_baseline_bpb=self.state.baseline_bpb,
                n_candidates=len(candidates),
                n_crashes=sum(1 for c in candidates if c.status == "crash"),
            )
            print("  -> nothing beat the baseline; lineage unchanged")

    # -- one candidate ------------------------------------------------------

    def _execute(self, ctx: phases.PhaseContext, agent: AgentSpec, variant: int, session: str) -> Candidate:
        worktree = self.trees.create(ctx.round_idx, agent.id, variant, self.state.baseline_commit)
        cand = Candidate(agent=agent.id, variant=variant, worktree=worktree)

        try:
            result = phases.run_finalize(
                self.harnesses[agent.id], agent, ctx, session, worktree, variant
            )
            cand.cost_usd = result.cost_usd
            cand.num_turns = result.num_turns
        except HarnessError as exc:
            cand.error = f"agent session failed: {exc}"

        meta = phases.read_candidate_json(worktree)
        cand.title = meta.get("title", "")
        cand.description = meta.get("description", "")
        cand.reflection = meta.get("reflection", "")

        metrics = parse_run_log(worktree / "run.log")
        cand.status = "ok" if metrics.ok else "crash"
        cand.val_bpb = metrics.val_bpb
        cand.peak_vram_mb = metrics.peak_vram_mb
        if not metrics.ok:
            cand.error = (cand.error + "\n" + metrics.error).strip()
        if meta.get("status") == "abandoned" and not metrics.ok:
            cand.status = "abandoned"

        run_record = read_arena_run(worktree)
        cand.attempts = run_record.get("runs_after", 0) if run_record else 0

        cand.commit = self.trees.commit_all(
            worktree, f"[{self.tag}] r{ctx.round_idx} {agent.id}/v{variant}: {cand.description[:60]}"
        )
        self.trees.keep(cand.commit, ctx.round_idx, agent.id, variant)
        self._archive(ctx.round_idx, agent.id, variant, worktree)
        self.trees.remove(worktree)
        return cand

    def _archive(self, round_idx: int, agent: str, variant: int, worktree: Path) -> None:
        dest = self.run_dir / "rounds" / f"r{round_idx:02d}" / f"{agent}_v{variant}"
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("run.log", "candidate.json", "arena_run.json", "train.py"):
            src = worktree / name
            if src.exists():
                shutil.copy2(src, dest / name)

    # -- teardown ------------------------------------------------------------

    def _finish(self) -> None:
        self.log.publish(upto_round=self.state.round_idx)
        self.trees.prune()

        overlaps = assert_no_overlap(self.run_dir / "gpu_timeline.jsonl")
        b = self.budget()
        summary = {
            "cell_id": self.cfg.cell_id,
            "rounds_completed": len(self.state.completed_rounds),
            "final_val_bpb": self.state.baseline_bpb,
            "gpu_seconds_spent": b["spent_s"],
            "gpu_seconds_budget": b["budget_s"],
            "training_runs": b["runs"],
            "agent_cost_usd": round(self.state.total_cost_usd, 4),
            "wall_clock_s": round((self.state.finished_at or time.time()) - self.state.started_at, 1),
            "stop_reason": self.state.stop_reason,
            "gpu_overlaps": len(overlaps),
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        print("\n" + "=" * 60)
        for k, v in summary.items():
            print(f"{k:>22}: {v}")
        if overlaps:
            print("\n!! GPU runs overlapped — this cell's timings are NOT comparable.")
