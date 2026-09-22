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
# Slots claimed by runs that were killed before recording anything are given back, so the cell
# ends with the full budget of real measurements. Recorded in budget.json as "reclaimed".
python3 - "$run_dir" <<'PY'
import json, sys, collections
d = sys.argv[1]
b = json.load(open(d + "/budget.json"))
tl = [json.loads(l) for l in open(d + "/gpu_timeline.jsonl")] if __import__("os").path.exists(d + "/gpu_timeline.jsonl") else []
# Match claims to recorded runs PER AGENT: open-protocol claims all carry round 0 while their
# records carry the run number, so a (agent, round) match is wrong there. Per agent, the number
# of claims beyond the number of recorded runs is what was lost.
rec = collections.Counter(r["agent"] for r in tl)
seen = collections.Counter(); kept, lost = [], []
for c in b.get("claims", []):
    a = c["agent"]; seen[a] += 1
    (kept if seen[a] <= rec[a] else lost).append(c)
if lost:
    b["claims"] = kept; b["runs"] = len(kept); b.setdefault("reclaimed", []).extend(lost)
    json.dump(b, open(d + "/budget.json", "w"), indent=2)
    print(f"reclaimed {len(lost)} lost slot(s); runs now {b['runs']} of {b['budget_runs']}")
else:
    print("no lost slots to reclaim")
PY
echo "=== $(date '+%F %T') resuming $cell from $run_dir ==="
for pass in 1 2 3; do
  python3 -m orchestrator.run --resume-run "$run_dir"
  recorded=$(wc -l < "$run_dir/gpu_timeline.jsonl" 2>/dev/null | tr -d ' ')
  budget=$(python3 -c "import json;print(json.load(open('$run_dir/budget.json'))['budget_runs'])")
  [ "${recorded:-0}" -ge "$budget" ] && break
  python3 - "$run_dir" <<'PY'
import json, sys, collections, os
d = sys.argv[1]; b = json.load(open(d + "/budget.json"))
tl = [json.loads(l) for l in open(d + "/gpu_timeline.jsonl")] if os.path.exists(d + "/gpu_timeline.jsonl") else []
# Match claims to recorded runs PER AGENT: open-protocol claims all carry round 0 while their
# records carry the run number, so a (agent, round) match is wrong there. Per agent, the number
# of claims beyond the number of recorded runs is what was lost.
rec = collections.Counter(r["agent"] for r in tl)
seen = collections.Counter(); kept, lost = [], []
for c in b.get("claims", []):
    a = c["agent"]; seen[a] += 1
    (kept if seen[a] <= rec[a] else lost).append(c)
if lost:
    b["claims"] = kept; b["runs"] = len(kept); b.setdefault("reclaimed", []).extend(lost)
    json.dump(b, open(d + "/budget.json", "w"), indent=2); print(f"pass: reclaimed {len(lost)} more lost slot(s)")
PY
  echo "=== recorded $recorded of $budget after pass $pass; resuming again ==="
done
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
