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
  python3 -m analysis.report "$run_dir" > /dev/null || true
  if python3 -m analysis.verify "$run_dir"; then dest="results/$cell"; else dest="results/$cell-INVALID"; fi
  rm -rf "$dest"; mkdir -p "$dest"
  # everything except the worktrees (already removed) and nothing large
  rsync -a --exclude 'work/' "$run_dir/" "$dest/" 2>/dev/null || cp -R "$run_dir/." "$dest/"
  git add "$dest" && git -c user.name="${GIT_AUTHOR_NAME:-arena-box}" -c user.email="${GIT_AUTHOR_EMAIL:-arena@local}" \
    commit -qm "results: $cell ($stamp)" && echo "archived $dest (committed locally; push in the morning)"
  echo "=== $(date '+%F %T') finished $cell ==="
done
echo "all cells done: $*"
