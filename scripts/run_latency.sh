#!/usr/bin/env bash
# Latency, peak memory and parameter count for every model at nuImages (1600x900) and KITTI (about 1242x375) resolution. Run it on an otherwise idle GPU.
#
#   DATA=$HOME/data RUNS=$HOME/runs bash scripts/run_latency.sh
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${DATA:-$HOME/data}"; RUNS="${RUNS:-$HOME/runs}"
mkdir -p "$RUNS/logs"
cd "$REPO"
nvidia-smi > "$RUNS/logs/nvidia-smi_latency_$(date +%Y%m%d_%H%M%S).txt"
MODELS=(tv:fasterrcnn_resnet50_fpn tv:retinanet_resnet50_fpn tv:fcos_resnet50_fpn
        tv:fasterrcnn_resnet50_fpn_v2 yolo:yolo11n.pt yolo:yolo11m.pt yolo:yolo11x.pt
        yolo:yolo11m.pt@1280
        dfine:n dfine:m
        dfine:x dfine:m@letterbox
        gdino:IDEA-Research/grounding-dino-tiny
        gdino:openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_grit_v3det
        gdino:iSEE-Laboratory/llmdet_tiny)
# one process per model: the official D-FINE code keeps configuration state
# across builds, so two D-FINE sizes cannot be loaded in the same process
status=0
for ds in nuimages kitti; do
  root=$([ $ds = kitti ] && echo "$DATA/kitti" || echo "$DATA/nuimages")
  for m in "${MODELS[@]}"; do
    python -m detbench.latency --models "$m" --native "$DATA/native/${ds}_val.json" --images "$root" \
      --out "$RUNS/latency_$ds.json" >> "$RUNS/logs/latency_$ds.log" 2>&1 || { echo "FAILED latency $m on $ds"; status=1; }
  done
done
echo "latency done"
exit $status
