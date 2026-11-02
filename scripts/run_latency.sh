#!/usr/bin/env bash
# Latency, peak memory and parameter count for every model, on an otherwise
# idle GPU, at nuImages (1600x900) and KITTI (about 1242x375) resolution.
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
python -m detbench.latency --models "${MODELS[@]}" --native "$DATA/native/nuimages_val.json" \
  --images "$DATA/nuimages" --out "$RUNS/latency_nuimages.json" > "$RUNS/logs/latency_nuimages.log" 2>&1
python -m detbench.latency --models "${MODELS[@]}" --native "$DATA/native/kitti_val.json" \
  --images "$DATA/kitti" --out "$RUNS/latency_kitti.json" > "$RUNS/logs/latency_kitti.log" 2>&1
echo "latency done"
