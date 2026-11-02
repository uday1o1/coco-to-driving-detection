#!/usr/bin/env bash
# Every analysis in results/, the qualitative panel and the figures, from
# stored detections. CPU only.
#
#   DATA=$HOME/data RUNS=$HOME/runs bash scripts/run_analysis.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${DATA:-$HOME/data}"; RUNS="${RUNS:-$HOME/runs}"
PROCS="${PROCS:-$(nproc)}"
cd "$REPO"
for s in gate reproduction main audit stability fusion subgroups finetune errors prompts; do
  echo "[$(date +%H:%M:%S)] $s"
  python scripts/analyze.py $s --runs "$RUNS" --data "$DATA" --procs "$PROCS"
done
[ -f "$RUNS/latency_nuimages.json" ] && cp "$RUNS/latency_nuimages.json" results/latency_nuimages.json
[ -f "$RUNS/latency_kitti.json" ] && cp "$RUNS/latency_kitti.json" results/latency_kitti.json
python scripts/qualitative.py --runs "$RUNS" --data "$DATA"
python scripts/figures.py
echo "[$(date +%H:%M:%S)] analysis done"
