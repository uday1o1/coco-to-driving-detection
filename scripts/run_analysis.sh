#!/usr/bin/env bash
# Every analysis in results/, the qualitative panel and the figures, from
# stored detections. CPU only.
#
#   DATA=$HOME/data RUNS=$HOME/runs bash scripts/run_analysis.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${DATA:-$HOME/data}"; RUNS="${RUNS:-$HOME/runs}"
# Each worker holds a dataset's ground truth and one detection file (about 3 GB
# for nuImages), so keep PROCS within memory / 3 GB.
PROCS="${PROCS:-12}"
SECTIONS="${SECTIONS:-gate reproduction main audit stability fusion subgroups finetune errors prompts contrasts robustness}"
cd "$REPO"
for s in $SECTIONS; do
  echo "[$(date +%H:%M:%S)] $s"
  python scripts/analyze.py $s --runs "$RUNS" --data "$DATA" --procs "$PROCS"
done
[ -f "$RUNS/latency_nuimages.json" ] && cp "$RUNS/latency_nuimages.json" results/latency_nuimages.json
[ -f "$RUNS/latency_kitti.json" ] && cp "$RUNS/latency_kitti.json" results/latency_kitti.json
if [ -d "$DATA/kitti/training/image_2" ] && [ -d "$DATA/nuimages/samples" ]; then
  python scripts/qualitative.py --runs "$RUNS" --data "$DATA"     # needs images
fi
python scripts/figures.py
echo "[$(date +%H:%M:%S)] analysis done"
