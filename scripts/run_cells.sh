#!/usr/bin/env bash
# Run cells one after another on this box, unattended, and archive each one.
#
#   nohup bash scripts/run_cells.sh open_haiku_6 indep_haiku_6 > runs/tonight.out 2>&1 &
#   tail -f runs/tonight.out
#
# For each cell: run it, verify the invariants, write the report, copy the run
# directory to results/<cell>/ and commit it locally.  Push results/ yourself
# in the morning (`git push`).  A cell that fails verification is still
# archived, under results/<cell>-INVALID/, so nothing is lost.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs results
# Claude Code refuses --dangerously-skip-permissions as root unless it believes it is in a sandbox;
# cloud GPU containers run as root, and the guard hook is what actually fences the agents in.
export IS_SANDBOX=1

[ -n "${ANTHROPIC_API_KEY:-}" ] || claude auth status 2>/dev/null | grep -q '"loggedIn": true' || {
  echo "no ANTHROPIC_API_KEY and claude is not logged in; agent calls would fail" >&2; exit 1; }
[ -f "${AUTORESEARCH_DIR:-$HOME/autoresearch}/baseline.json" ] || {
  echo "no baseline.json: run scripts/setup_gpu_box.sh first" >&2; exit 1; }

for cell in "$@"; do
  cfg="configs/$cell.yaml"
  [ -f "$cfg" ] || { echo "no such config: $cfg" >&2; continue; }
  stamp=$(date +%Y%m%d-%H%M%S)
  run_dir="runs/$cell-$stamp"
  echo "=== $(date '+%F %T') starting $cell -> $run_dir ==="
  python3 -m orchestrator.run --config "$cfg" --run-dir "$run_dir"
  # Agents sometimes launch arena-train in the background and then report "finished"; their jobs
  # would keep claiming slots and using the GPU after the cell is over. Kill everything that still
  # carries this cell's ARENA_RUN_DIR before archiving and before the next cell starts.
  abs_run_dir=$(cd "$run_dir" && pwd)
  for p in $(ps -eo pid=); do
    d=$(tr "\0" "\n" < /proc/$p/environ 2>/dev/null | grep "^ARENA_RUN_DIR=" | cut -d= -f2)
    [ "$d" = "$abs_run_dir" ] && kill -9 "$p" 2>/dev/null
  done
  sleep 2
  python3 -m analysis.report "$run_dir" > /dev/null || true
  if python3 -m analysis.verify "$run_dir"; then dest="results/$cell"; else dest="results/$cell-INVALID"; fi
  rm -rf "$dest"; mkdir -p "$dest"
  # everything except the worktrees (already removed) and nothing large
  rsync -a --exclude 'work/' "$run_dir/" "$dest/" 2>/dev/null || cp -R "$run_dir/." "$dest/"
  # The full per-session traces (every turn, tool call and edit of every agent), which Claude Code
  # writes under ~/.claude/projects; keep the ones written since this cell started, for case studies.
  mkdir -p "$dest/claude_sessions"
  find "$HOME/.claude/projects" -name '*.jsonl' -newer "$run_dir/provenance.json" 2>/dev/null | while read -r f; do
    cp "$f" "$dest/claude_sessions/$(basename "$(dirname "$f")")__$(basename "$f")"
  done
  echo "archived $(ls "$dest/claude_sessions" | wc -l | tr -d ' ') session trace(s)"
  git add "$dest" && git -c user.name="${GIT_AUTHOR_NAME:-arena-box}" -c user.email="${GIT_AUTHOR_EMAIL:-arena@local}" \
    commit -qm "results: $cell ($stamp)" && echo "archived $dest (committed locally; push in the morning)"
  echo "=== $(date '+%F %T') finished $cell ==="
done
echo "all cells done: $*"
