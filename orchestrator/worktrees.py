"""Per-candidate git worktrees over the autoresearch repo.

Every candidate in a round needs its own checkout of ``train.py`` to hack on,
and the winner's commit has to survive to become the next round's baseline.
Worktrees give both cheaply: one clone, N isolated working directories, and a
ref per candidate so nothing is lost to garbage collection.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


class GitError(RuntimeError):
    pass


def git(repo: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
    )
    if check and proc.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed in {repo}:\n{proc.stderr.strip()}")
    return proc.stdout.strip()


class WorktreeManager:
    def __init__(self, repo: Path, run_dir: Path, tag: str):
        self.repo = Path(repo).expanduser().resolve()
        self.run_dir = Path(run_dir)
        self.tag = tag
        self.work_root = self.run_dir / "work"
        self.work_root.mkdir(parents=True, exist_ok=True)
        if not (self.repo / ".git").exists():
            raise GitError(f"{self.repo} is not a git repository")

    # -- commits ---------------------------------------------------------

    def resolve(self, rev: str) -> str:
        return git(self.repo, "rev-parse", rev)

    def head_commit(self, worktree: Path) -> str:
        return git(worktree, "rev-parse", "HEAD")

    def is_dirty(self, worktree: Path) -> bool:
        return bool(git(worktree, "status", "--porcelain"))

    def commit_all(self, worktree: Path, message: str) -> str:
        """Capture whatever the agent left behind, committed or not."""
        if self.is_dirty(worktree):
            git(worktree, "add", "-A")
            git(worktree, "-c", "user.name=arena", "-c", "user.email=arena@local",
                "commit", "-m", message, check=False)
        return self.head_commit(worktree)

    def diff(self, base: str, rev: str, path: str = "train.py") -> str:
        return git(self.repo, "diff", f"{base}..{rev}", "--", path, check=False)

    # -- worktrees --------------------------------------------------------

    def create(self, round_idx: int, agent: str, variant: int, base_commit: str) -> Path:
        name = f"r{round_idx:02d}_{agent}_v{variant}"
        path = self.work_root / name
        if path.exists():
            self.remove(path)
        git(self.repo, "worktree", "add", "--detach", str(path), base_commit)
        return path

    def create_named(self, name: str, base_commit: str) -> Path:
        """A long-lived worktree with a fixed name (the open protocol's private scratch)."""
        path = self.work_root / name
        if path.exists():
            self.remove(path)
        git(self.repo, "worktree", "add", "--detach", str(path), base_commit)
        return path

    def keep(self, commit: str, round_idx: int, agent: str, variant: int) -> str:
        """Pin a candidate commit behind a ref so it is never gc'd."""
        ref = f"refs/arena/{self.tag}/r{round_idx:02d}_{agent}_v{variant}"
        git(self.repo, "update-ref", ref, commit)
        return ref

    def remove(self, path: Path) -> None:
        git(self.repo, "worktree", "remove", "--force", str(path), check=False)
        if Path(path).exists():
            subprocess.run(["rm", "-rf", str(path)], check=False)

    def prune(self) -> None:
        git(self.repo, "worktree", "prune", check=False)
