"""The open protocol: asynchronous agents over a shared directory.

The round loop in ``arena.py`` is one arm of the experiment.  This is the
other: the communication structure of Park, Kontonis, Garg, Krishnamurthy and
Papailiopoulos, "Scaling Discovery through Test-Time Communication" (arXiv
2609.21032, Sep 2026).  No rounds and no roles.  Every agent gets one long
autonomous session in a private worktree, a share of the cell's run budget,
and read/write access to a shared directory with a fixed layout:

    slots/<agent>/approach.md   the approach it declared (distinct by rule)
    findings                    append-only broadcast, with commit pointers
    disconfirmations            negative results
    score log                   one line per attempt, written by arena-train
    coordination                conventions agreed after a collision
    worktree                    private scratch, never shared

The paper's two behavioural rules are carried by the prompt, not by code: an
agent adopts a peer's approach only after observing a clearly better measured
result (``arena-adopt`` puts that on the record), and keeps one meaningful
variation of its own afterwards.

Compute matching is unchanged: ``arena-train`` claims from the same run budget
as the round protocol, so an open cell and a round cell with the same
``train_run_budget`` are directly comparable.  ``open_share_log: false`` runs
the same agents blind to each other -- the paper's independent (best@k)
control, at matched compute.

A session that returns early (turn cap, dollar cap, or the model deciding it
is done) is resumed while the agent still has runs in its share, up to
``open_max_resumes`` times.  Progress is on disk after every run, so a killed
orchestrator resumes the cell where it stopped.
"""

from __future__ import annotations

import json
import shutil
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import phases
from .arena import Arena
from .config import AgentSpec
from .gpu import assert_no_overlap
from .harness import HarnessError
from .sharedlog import open_log_path

OPEN_TOOLS = ("Bash", "Read", "Edit", "Write", "Grep", "Glob")


class OpenArena(Arena):
    """Reuses Arena's setup (baseline, budget, provenance, worktrees, harnesses)."""

    def __init__(self, cfg, run_dir: Path, resume: bool = False):
        super().__init__(cfg, run_dir, resume=resume)
        (self.run_dir / "slots").mkdir(exist_ok=True)
        (self.run_dir / "runs").mkdir(exist_ok=True)
        if not resume:
            b = json.loads((self.run_dir / "budget.json").read_text())
            b["per_agent_quota"] = self.cfg.per_agent_quota
            b["protocol"] = "open"
            (self.run_dir / "budget.json").write_text(json.dumps(b, indent=2))
            self.log.append(
                "round_start", round=0,
                baseline_commit=self.state.baseline_commit,
                baseline_bpb=self.state.baseline_bpb,
                runs_remaining=self.cfg.train_run_budget,
                protocol="open", share_log=self.cfg.open_share_log,
            )
        self.sessions_path = self.run_dir / "open_sessions.json"
        self.sessions: dict[str, dict] = (
            json.loads(self.sessions_path.read_text()) if self.sessions_path.exists() else {}
        )

    # -- helpers ------------------------------------------------------------

    def _agent_ids(self) -> list[str]:
        return [a.id for a in self.cfg.agents]

    def _publish(self) -> None:
        self.log.publish_open(budget=self.budget(), agents=self._agent_ids(),
                              share=self.cfg.open_share_log)

    def _runs_used_by(self, agent: str) -> int:
        return sum(1 for c in self.budget().get("claims", []) if c.get("agent") == agent)

    def _share_left(self, agent: str) -> int:
        quota = self.cfg.per_agent_quota
        if quota:
            return max(0, quota - self._runs_used_by(agent))
        return self.runs_left()

    def _env(self, agent: AgentSpec) -> dict[str, str]:
        env = phases.agent_env(self._context(0), agent, variant=0, phase="open")
        env.update({
            "ARENA_PROTOCOL": "open",
            "ARENA_SHARE_LOG": "1" if self.cfg.open_share_log else "0",
            "ARENA_AGENTS": ",".join(self._agent_ids()),
            "ARENA_AGENT_QUOTA": str(self.cfg.per_agent_quota),
            "ARENA_TAG": self.tag,
        })
        return env

    def _worktree(self, agent: AgentSpec) -> Path:
        path = self.trees.work_root / f"open_{agent.id}"
        if not path.exists():
            self.trees.create_named(f"open_{agent.id}", self.state.baseline_commit)
        return path

    def _system_prompt(self, agent: AgentSpec) -> str:
        peers = [a for a in self._agent_ids() if a != agent.id]
        log_path = open_log_path(self.run_dir, agent.id, self.cfg.open_share_log)
        return phases.render(
            "system_open.md",
            AGENT_ID=agent.id,
            N_AGENTS=str(self.cfg.n_agents),
            PEER_IDS=", ".join(peers) if peers else "none — you are working alone this cell",
            SHARING=("on: every finding, disconfirmation and score is visible to all agents"
                     if self.cfg.open_share_log else
                     "OFF: this cell runs agents independently; you see only your own entries"),
            LOG_PATH=str(log_path),
            RESULTS_PATH=str(self.log.results_tsv),
            QUOTA=str(self.cfg.per_agent_quota or self.cfg.train_run_budget),
            CELL_BUDGET=str(self.cfg.train_run_budget),
        )

    # -- one agent, start to finish -------------------------------------------

    def _run_agent(self, agent: AgentSpec) -> dict:
        worktree = self._worktree(agent)
        info = self.sessions.setdefault(agent.id, {"session_id": "", "segments": 0, "cost_usd": 0.0,
                                                   "done": False})
        harness = self.harnesses[agent.id]
        log_path = open_log_path(self.run_dir, agent.id, self.cfg.open_share_log)

        while not info["done"]:
            left = self._share_left(agent.id)
            if left <= 0 or self.runs_left() <= 0:
                info["done"] = True
                info["stop"] = "share spent"
                break
            if info["segments"] > self.cfg.open_max_resumes:
                info["done"] = True
                info["stop"] = "resume cap reached"
                break
            if self.state.total_cost_usd >= self.cfg.cell_budget_usd:
                info["done"] = True
                info["stop"] = "cell dollar ceiling"
                break

            first = info["segments"] == 0
            template = "open.md" if first else "open_continue.md"
            prompt = phases.render(
                template,
                WORKTREE=str(worktree),
                RUNS_LEFT=str(left),
                CELL_LEFT=str(self.runs_left()),
                LOG_MD=log_path.read_text() if log_path.exists() else "",
                LOG_PATH=str(log_path),
            )
            seg = info["segments"]
            print(f"  {agent.id}: session segment {seg} ({left} runs in share) ...", flush=True)
            try:
                result = harness.query(
                    prompt,
                    cwd=worktree,
                    model=agent.model,
                    effort=agent.effort,
                    schema=None,
                    tools=OPEN_TOOLS,
                    resume=info["session_id"] or None,
                    fork=False,
                    session_id=None if info["session_id"] else phases.new_session_id(),
                    permission_mode="bypassPermissions",
                    phase="open",
                    timeout_s=self.cfg.open_session_timeout_s,
                    append_system_prompt=self._system_prompt(agent),
                    settings=self.settings_path,
                    add_dirs=[self.run_dir],
                    env=self._env(agent),
                    max_budget_usd=self.cfg.max_budget_usd_per_session,
                    transcript_path=self.run_dir / "transcripts" / f"open_{agent.id}_seg{seg:02d}.json",
                )
                info["session_id"] = result.session_id or info["session_id"]
                info["cost_usd"] += result.cost_usd
                self._charge(result.cost_usd)
                self.log.append("session_segment", round=0, agent=agent.id, segment=seg,
                                cost_usd=result.cost_usd, num_turns=result.num_turns,
                                is_error=result.is_error, duration_s=round(result.duration_s, 1))
                # A segment that errored out almost immediately is a rate limit or an
                # auth failure, not the agent stopping: wait, and do not count it as a
                # resume.  Six Opus sessions on a claude.ai plan hit the 5-hour window;
                # the window passes, the cell continues.
                if result.is_error and result.num_turns <= 1:
                    info["throttled"] = info.get("throttled", 0) + 1
                    if info["throttled"] > 36:  # 6 hours of waiting: something else is wrong
                        info["done"] = True
                        info["stop"] = f"gave up after {info['throttled']} errored segments: {result.text[:120]}"
                        break
                    print(f"  ~ {agent.id}: segment errored at once ({result.text[:80]}); waiting 10 min")
                    time.sleep(600)
                    continue
            except HarnessError as exc:
                self.log.append("session_segment", round=0, agent=agent.id, segment=seg,
                                error=str(exc)[:1000])
                print(f"  ! {agent.id}: segment {seg} failed: {exc}")
                info["throttled"] = info.get("throttled", 0) + 1
                if info["throttled"] > 36:
                    info["done"] = True
                    info["stop"] = "gave up after repeated harness failures"
                    break
                time.sleep(120)
                continue
            info["segments"] += 1
            self._save_sessions()
            self.state.save(self.run_dir)

            final = worktree / "final.json"
            if final.exists():
                try:
                    if json.loads(final.read_text()).get("status") == "finished":
                        info["done"] = True
                        info["stop"] = "agent finished"
                except json.JSONDecodeError:
                    pass

        # pin whatever the agent left behind
        commit = self.trees.commit_all(worktree, f"[{self.tag}] open {agent.id}: final state")
        self.trees.keep(commit, 99, agent.id, 0)
        dest = self.run_dir / "runs" / agent.id
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("final.json", "train.py"):
            if (worktree / name).exists():
                shutil.copy2(worktree / name, dest / name)
        info["final_commit"] = commit
        self._save_sessions()
        return info

    def _save_sessions(self) -> None:
        self.sessions_path.write_text(json.dumps(self.sessions, indent=2))

    # -- the cell ---------------------------------------------------------------

    def run(self):
        print(self.cfg.summary())
        self._publish()
        with ThreadPoolExecutor(max_workers=self.cfg.n_agents) as pool:
            futures = {pool.submit(self._run_agent, a): a.id for a in self.cfg.agents}
            for fut in as_completed(futures):
                aid = futures[fut]
                try:
                    info = fut.result()
                    print(f"  {aid}: done ({info.get('stop', '?')}, {info['segments']} segment(s), "
                          f"${info['cost_usd']:.2f})")
                except Exception:
                    with (self.run_dir / "errors.log").open("a") as fh:
                        fh.write(f"\n=== open {aid} ===\n{traceback.format_exc()}\n")
                    print(f"  ! {aid}: orchestration failed; see errors.log")

        # the cell's result is the best measured run, whoever made it
        best = None
        for c in self.log.records():
            if c.get("t") == "candidate" and c.get("status") == "ok" and c.get("val_bpb") is not None:
                if best is None or c["val_bpb"] < best["val_bpb"]:
                    best = c
        if best:
            self.state.baseline_bpb = best["val_bpb"]
            self.state.baseline_commit = best.get("commit", self.state.baseline_commit)
        self.log.append("round_end", round=0, improved=bool(best),
                        winner={"agent": best["agent"], "variant": 0, "commit": best.get("commit", "")} if best else None,
                        new_baseline_bpb=self.state.baseline_bpb,
                        n_candidates=sum(1 for c in self.log.records() if c.get("t") == "candidate"),
                        n_crashes=sum(1 for c in self.log.records()
                                      if c.get("t") == "candidate" and c.get("status") != "ok"),
                        runs_remaining=self.runs_left())
        self.state.round_idx = 1
        self.state.completed_rounds = [0]
        self.state.stop_reason = "all agents finished"
        self.state.finished_at = time.time()
        self.state.save(self.run_dir)
        self._publish()
        self.trees.prune()
        # leave the per-agent worktrees removed, like the round loop does
        for a in self.cfg.agents:
            wt = self.trees.work_root / f"open_{a.id}"
            if wt.exists():
                self.trees.remove(wt)
        overlaps = assert_no_overlap(self.run_dir / "gpu_timeline.jsonl")
        b = self.budget()
        summary = {
            "cell_id": self.cfg.cell_id,
            "protocol": "open",
            "share_log": self.cfg.open_share_log,
            "rounds_completed": 1,
            "final_val_bpb": self.state.baseline_bpb,
            "training_runs": b["runs"],
            "training_run_budget": b["budget_runs"],
            "gpu_seconds_spent": round(b.get("spent_s", 0.0), 1),
            "agent_cost_usd": round(self.state.total_cost_usd, 4),
            "wall_clock_s": round(self.state.finished_at - self.state.started_at, 1),
            "stop_reason": self.state.stop_reason,
            "gpu_overlaps": len(overlaps),
            "per_agent": {a: {"runs": self._runs_used_by(a), **{k: v for k, v in s.items()
                              if k in ("segments", "cost_usd", "stop")}}
                          for a, s in self.sessions.items()},
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print("\n" + "=" * 60)
        for k, v in summary.items():
            print(f"{k:>22}: {v}")
        if overlaps:
            print("\n!! GPU runs overlapped — this cell's timings are NOT comparable.")
        return self.state
