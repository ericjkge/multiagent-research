#!/usr/bin/env python3
"""Multi-agent autoresearch orchestrator.

One cell = one config (N agents, models, rounds). Each agent owns a git worktree of the autoresearch fork.
Each round: PROPOSE (all agents, parallel) -> RESPOND (all, parallel) -> FINALIZE+EDIT (all, parallel)
-> the orchestrator runs every agent's train.py sequentially on the GPU -> results go to the shared log.
Solo cells (1 agent) use ideas_per_agent > 1 (best-of-k) for compute matching: the agent proposes k ideas
and implements each in turn; all k are run.

Usage:
  python orchestrator/run.py --cell configs/cells.json:haiku_3 --out runs/haiku_3
  python orchestrator/run.py --cell configs/cells.json:haiku_3 --out runs/test --fake-gpu --rounds 2
  python orchestrator/run.py --cell configs/cells.json:haiku_3 --out runs/test --fake-gpu --fake-agents --rounds 2

Needs: autoresearch cloned at --base (default work/autoresearch) with `uv run prepare.py` already done
on a GPU box (the fake GPU mode skips that).
"""
import argparse
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import agents as A  # noqa: E402
import prompts as P  # noqa: E402


def sh(cmd, cwd=None, check=True, timeout=None):
    p = subprocess.run(cmd, cwd=cwd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout)
    if check and p.returncode != 0:
        raise RuntimeError(f"cmd failed ({p.returncode}): {cmd}\n{p.stderr[-2000:]}")
    return p.stdout.strip()


def now():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Cell:
    def __init__(self, cfg, out, base, fake_gpu, fake_agents):
        self.cfg = cfg
        self.out = out
        self.base = os.path.abspath(base)
        self.fake_gpu = fake_gpu
        self.fake_agents = fake_agents
        self.n = len(cfg["agents"])
        self.k = cfg.get("ideas_per_agent", 1)
        self.rounds = cfg["rounds"]
        self.sync_to_best = cfg.get("sync_to_best", True)
        os.makedirs(out, exist_ok=True)
        self.results_path = os.path.join(out, "results.jsonl")
        self.log_path = os.path.join(out, "log.md")
        self.results = []
        if os.path.exists(self.results_path):
            self.results = [json.loads(l) for l in open(self.results_path) if l.strip()]
        self.worktrees = {}
        self.best = None  # (val_bpb, commit, agent)

    # ---------- setup ----------
    def setup(self):
        if not os.path.isdir(os.path.join(self.base, ".git")):
            raise SystemExit(f"autoresearch not found at {self.base}; clone it first (see RUNBOOK.md)")
        base_commit = sh("git rev-parse --short HEAD", cwd=self.base)
        for i, ag in enumerate(self.cfg["agents"]):
            aid = ag["id"]
            wt = os.path.abspath(os.path.join(self.out, "work", aid))
            if not os.path.isdir(wt):
                branch = f"{os.path.basename(self.out)}_{aid}"
                sh("git worktree prune", cwd=self.base, check=False)
                sh(f"git branch -D {branch}", cwd=self.base, check=False)  # stale branch from a deleted run dir
                sh(f"git worktree add -f {wt} -b {branch} {base_commit}", cwd=self.base)
            self.worktrees[aid] = wt
        if not self.results:
            self._log(f"# Cell `{self.cfg['name']}` started {now()}\n\n"
                      f"agents: {[a['id'] + ':' + a['model'] for a in self.cfg['agents']]}, rounds: {self.rounds}, "
                      f"ideas per agent: {self.k}, base commit {base_commit}\n")
            # round 0: baseline, run once
            r = self._train(self.worktrees[self.cfg["agents"][0]["id"]])
            self._record(0, "baseline", "baseline (unmodified train.py)", r, base_commit, status="keep", predicted=None)
        self._refresh_best()

    # ---------- main loop ----------
    def run(self):
        done_rounds = max([r["round"] for r in self.results] + [0])
        for rnd in range(done_rounds + 1, self.rounds + 1):
            t0 = time.time()
            self._log(f"\n## Round {rnd}  ({now()})\n")
            if self.sync_to_best and self.best:
                self._sync_all_to_best()
            if self.n == 1 and self.k > 1:
                self._solo_round(rnd)
            else:
                self._collab_round(rnd)
            self._refresh_best()
            self._log(f"\nRound {rnd} done in {round((time.time() - t0) / 60, 1)} min. "
                      f"Global best val_bpb {self.best[0]:.6f} (agent {self.best[2]}, commit {self.best[1]}).\n")

    def _collab_round(self, rnd):
        log = self._log_text()
        # step 1: propose
        props = self._parallel(lambda ag: self._ask(ag, P.PROPOSE, rnd, log=log, allow_edits=False))
        props_txt = "\n".join(f"- {aid}: {j.get('idea', '?')} (why: {j.get('justification', '')})" for aid, j in props.items())
        self._log("Proposals:\n" + props_txt + "\n")
        # step 2: respond
        resps = self._parallel(lambda ag: self._ask(ag, P.RESPOND, rnd, log=log, proposals=props_txt, allow_edits=False))
        resps_txt = "\n".join(f"- {aid}: {j.get('response', '')}" for aid, j in resps.items())
        self._log("Responses:\n" + resps_txt + "\n")
        # step 3: finalize + edit (parallel, each in own worktree)
        finals = self._parallel(lambda ag: self._ask(ag, P.FINALIZE, rnd, log=log, proposals=props_txt,
                                                     responses=resps_txt, allow_edits=True))
        # step 4: run each agent's train.py sequentially on the GPU
        for ag in self.cfg["agents"]:
            aid = ag["id"]
            j = finals.get(aid) or {}
            idea = j.get("final_idea", props.get(aid, {}).get("idea", "?"))
            self._run_and_record(rnd, aid, idea, j.get("predicted_val_bpb"), proposal=props.get(aid, {}).get("idea"),
                                 changed=j.get("changed_from_proposal"))

    def _solo_round(self, rnd):
        ag = self.cfg["agents"][0]
        aid = ag["id"]
        log = self._log_text()
        j = self._ask(ag, P.SOLO_PROPOSE, rnd, log=log, k=self.k, allow_edits=False)
        ideas = [x.get("idea", "?") for x in (j.get("ideas") or [])][: self.k]
        while len(ideas) < self.k:
            ideas.append("(agent returned fewer ideas than requested)")
        self._log("Proposals:\n" + "\n".join(f"- {aid}: {i}" for i in ideas) + "\n")
        wt = self.worktrees[aid]
        start = sh("git rev-parse --short HEAD", cwd=wt)
        first = len(self.results)
        for idx, idea in enumerate(ideas):
            sh(f"git reset --hard {start}", cwd=wt)
            jj = self._ask(ag, P.SOLO_IMPLEMENT, rnd, log=self._log_text(), idea=idea, allow_edits=True)
            self._run_and_record(rnd, f"{aid}.{idx + 1}", idea, jj.get("predicted_val_bpb"), wt_override=wt, keep_rule="none")
        # best-of-k: the branch advances to the best idea of the round if it beats the agent's best so far
        rows = self.results[first:]
        scored = [x for x in rows if x.get("val_bpb") is not None]
        ref = self._agent_best_before(aid, first)
        if scored:
            b = min(scored, key=lambda x: x["val_bpb"])
            if ref is None or b["val_bpb"] < ref:
                b["status"] = "keep"
                sh(f"git reset --hard {b['commit']}", cwd=wt)
                self._rewrite_results()
                self._log(f"- best-of-{self.k}: kept {b['agent']} ({b['val_bpb']:.6f})\n")
                return
        sh(f"git reset --hard {start}", cwd=wt)
        self._log(f"- best-of-{self.k}: nothing beat {ref}; branch stays at {start}\n")

    # ---------- agent calls ----------
    def _ask(self, ag, template, rnd, allow_edits, **kw):
        aid = ag["id"]
        prompt = template.format(n_agents=self.n, agent_id=aid, round=rnd, n_rounds=self.rounds, **kw)
        kind = "fake" if self.fake_agents else ag.get("kind", "claude")
        try:
            text, meta = A.run_agent(kind, ag["model"], self.worktrees[aid], prompt, allow_edits=allow_edits,
                                     timeout=ag.get("timeout", 900), effort=ag.get("effort"))
        except Exception as e:  # timeouts etc. consume the step
            text, meta = "", {"error": str(e)[:500]}
        j = A.parse_json(text) or {}
        with open(os.path.join(self.out, "agent_calls.jsonl"), "a") as f:
            f.write(json.dumps({"t": now(), "round": rnd, "agent": aid, "step": template[:40].split("\n")[0],
                                "allow_edits": allow_edits, "meta": meta, "parsed": j, "raw_tail": text[-800:]}) + "\n")
        return j

    def _parallel(self, fn):
        out = {}
        with cf.ThreadPoolExecutor(max_workers=self.n) as ex:
            futs = {ex.submit(fn, ag): ag["id"] for ag in self.cfg["agents"]}
            for fu, aid in futs.items():
                try:
                    out[aid] = fu.result() or {}
                except Exception as e:
                    out[aid] = {"error": str(e)[:300]}
        return out

    # ---------- training ----------
    def _run_and_record(self, rnd, aid, idea, predicted, wt_override=None, keep_rule="agent", proposal=None, changed=None):
        base_aid = aid.split(".")[0]
        wt = wt_override or self.worktrees[base_aid]
        changed_files = sh("git status --porcelain", cwd=wt)
        bad = [l for l in changed_files.splitlines() if not l.endswith("train.py")]
        if bad:  # agent touched something else: revert those, keep train.py
            sh("git checkout -- . ':!train.py'", cwd=wt, check=False)
            sh("git clean -fdq", cwd=wt, check=False)
        prev = sh("git rev-parse --short HEAD", cwd=wt)
        if changed_files:
            sh("git add train.py", cwd=wt)
            sh(f'git commit -qm "r{rnd} {aid}: {idea[:60].replace(chr(34), "")}"', cwd=wt)
        commit = sh("git rev-parse --short HEAD", cwd=wt)
        no_change = (commit == prev)
        r = self._train(wt) if not no_change else {"val_bpb": None, "status": "nochange"}
        ref = self._agent_best(base_aid) if keep_rule == "agent" else (self.best[0] if self.best else None)
        if r.get("val_bpb") is None:
            status = r.get("status", "crash")
            if keep_rule != "none":
                sh(f"git reset --hard {prev}", cwd=wt)
        elif keep_rule == "none":
            status = "candidate"  # solo best-of-k decides after all k are run
        elif ref is None or r["val_bpb"] < ref:
            status = "keep"
        else:
            status = "discard"
            sh(f"git reset --hard {prev}", cwd=wt)
        self._record(rnd, aid, idea, r, commit, status, predicted, proposal=proposal, changed=changed)

    def _train(self, wt):
        if self.fake_gpu:
            return self._fake_train(wt)
        log_path = os.path.join(wt, "run.log")
        try:
            with open(log_path, "w") as f:
                p = subprocess.run(["uv", "run", "train.py"], cwd=wt, stdout=f, stderr=subprocess.STDOUT,
                                   timeout=self.cfg.get("train_timeout", 1500))
        except subprocess.TimeoutExpired:
            return {"val_bpb": None, "status": "timeout"}
        txt = open(log_path, errors="ignore").read()
        m = re.search(r"^val_bpb:\s*([0-9.]+)", txt, re.M)
        if p.returncode != 0 or not m:
            return {"val_bpb": None, "status": "crash", "tail": txt[-600:]}
        g = lambda k: (re.search(rf"^{k}:\s*([0-9.]+)", txt, re.M) or [None, None])[1]
        return {"val_bpb": float(m.group(1)), "status": "ok", "peak_vram_mb": g("peak_vram_mb"),
                "num_steps": g("num_steps"), "num_params_M": g("num_params_M"), "mfu": g("mfu_percent")}

    def _fake_train(self, wt):
        src = open(os.path.join(wt, "train.py")).read()
        if "raise" in sh("git diff HEAD~1 -- train.py", cwd=wt, check=False):
            return {"val_bpb": None, "status": "crash"}
        h = int(hashlib.md5(src.encode()).hexdigest(), 16) % 1000 / 1000
        time.sleep(0.2)
        return {"val_bpb": round(0.985 + 0.03 * h + random.gauss(0, 0.002), 6), "status": "ok", "num_steps": "fake"}

    # ---------- bookkeeping ----------
    def _record(self, rnd, aid, idea, r, commit, status, predicted, proposal=None, changed=None):
        row = {"t": now(), "round": rnd, "agent": aid, "commit": commit, "idea": idea, "proposal": proposal,
               "changed_from_proposal": changed, "predicted_val_bpb": predicted, **r,
               "status": status, "run_status": r.get("status")}
        self.results.append(row)
        with open(self.results_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        vb = f"{r['val_bpb']:.6f}" if r.get("val_bpb") is not None else status.upper()
        self._log(f"- result r{rnd} {aid} [{status}] val_bpb={vb} commit={commit} idea: {idea}\n")

    def _agent_best(self, aid):
        return self._agent_best_before(aid, len(self.results))

    def _agent_best_before(self, aid, upto):
        v = [x["val_bpb"] for x in self.results[:upto] if x.get("val_bpb") is not None and x["status"] == "keep"
             and (x["agent"].split(".")[0] == aid or x["agent"] == "baseline")]
        return min(v) if v else None

    def _rewrite_results(self):
        with open(self.results_path, "w") as f:
            for x in self.results:
                f.write(json.dumps(x) + "\n")

    def _refresh_best(self):
        ok = [x for x in self.results if x.get("val_bpb") is not None and x["status"] == "keep"]
        if ok:
            b = min(ok, key=lambda x: x["val_bpb"])
            self.best = (b["val_bpb"], b["commit"], b["agent"])
        json.dump({"best": self.best, "n_results": len(self.results)}, open(os.path.join(self.out, "state.json"), "w"))

    def _sync_all_to_best(self):
        """Start every agent's round from the global best train.py (shared repo semantics)."""
        _, commit, _ = self.best
        for aid, wt in self.worktrees.items():
            head = sh("git rev-parse --short HEAD", cwd=wt)
            if head != commit:
                sh(f"git fetch -q . 2>/dev/null; git checkout -q {commit} -- train.py", cwd=wt, check=False)
                if sh("git status --porcelain", cwd=wt):
                    sh(f'git commit -qam "sync to global best {commit}"', cwd=wt)

    def _log(self, s):
        with open(self.log_path, "a") as f:
            f.write(s)

    def _log_text(self, max_chars=24000):
        t = open(self.log_path).read() if os.path.exists(self.log_path) else ""
        return P.LOG_INTRO + (t if len(t) <= max_chars else "...(older rounds truncated)...\n" + t[-max_chars:])


def load_cell(spec, overrides):
    path, _, name = spec.partition(":")
    cells = json.load(open(path))
    cfg = dict(cells[name]) if name else dict(cells)
    cfg["name"] = name or cfg.get("name", "cell")
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, help="configs/cells.json:<name>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="work/autoresearch")
    ap.add_argument("--rounds", type=int)
    ap.add_argument("--fake-gpu", action="store_true", help="no training; hash-based fake score")
    ap.add_argument("--fake-agents", action="store_true", help="no API calls; canned agent output")
    a = ap.parse_args()
    cfg = load_cell(a.cell, {"rounds": a.rounds})
    cell = Cell(cfg, a.out, a.base, a.fake_gpu, a.fake_agents)
    cell.setup()
    cell.run()
    print(json.dumps({"cell": cfg["name"], "best": cell.best, "results": len(cell.results)}))
