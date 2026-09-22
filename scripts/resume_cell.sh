#!/usr/bin/env bash
# Continue a cell that stopped short from its saved state, then verify and archive it.
#   scripts/resume_cell.sh runs/opus_6-20260922-070031            # plain resume
#   scripts/resume_cell.sh runs/opus_1-2026... '{"bon": 3}'        # patch the cell config first
# The patch (JSON) is merged into provenance.json's "cell" block, e.g. to lift a cost cap
# or change best-of-N so the remaining budget divides into whole rounds. It is recorded there.
set -uo pipefail
cd "$(dirname "$0")/.."
export IS_SANDBOX=1
run_dir=$1; patch=${2:-}
cell=$(basename "$run_dir" | sed -E 's/-2026.*//')
if [ -n "$patch" ]; then
  python3 - "$run_dir/provenance.json" "$patch" <<'PY'
import json, sys
p, patch = sys.argv[1], json.loads(sys.argv[2])
d = json.load(open(p)); d["cell"].update(patch); d.setdefault("resume_patches", []).append(patch)
json.dump(d, open(p, "w"), indent=2); print("provenance patched:", patch)
PY
fi
echo "=== $(date '+%F %T') resuming $cell from $run_dir ==="
python3 -m orchestrator.run --resume-run "$run_dir"
abs_run_dir=$(cd "$run_dir" && pwd)
for p in $(ps -eo pid=); do d=$(tr "\0" "\n" < /proc/$p/environ 2>/dev/null | grep "^ARENA_RUN_DIR=" | cut -d= -f2); [ "$d" = "$abs_run_dir" ] && kill -9 "$p" 2>/dev/null; done
sleep 2
python3 -m analysis.report "$run_dir" > /dev/null || true
if python3 -m analysis.verify "$run_dir"; then dest="results/$cell"; else dest="results/$cell-INVALID"; fi
rm -rf "results/$cell" "results/$cell-INVALID"; mkdir -p "$dest"
rsync -a --exclude 'work/' "$run_dir/" "$dest/" 2>/dev/null || cp -R "$run_dir/." "$dest/"
mkdir -p "$dest/claude_sessions"
find "$HOME/.claude/projects" -name '*.jsonl' -newer "$run_dir/provenance.json" 2>/dev/null | while read -r f; do cp "$f" "$dest/claude_sessions/$(basename "$(dirname "$f")")__$(basename "$f")"; done
git add results && git -c user.name="arena-box" -c user.email="arena@local" commit -qm "results: $cell (resumed)" && echo "archived $dest"
echo "=== $(date '+%F %T') finished $cell ==="
