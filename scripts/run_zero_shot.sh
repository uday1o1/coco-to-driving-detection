#!/usr/bin/env bash
# Every zero-shot inference run in the study. Skips runs whose output exists,
# so it can be restarted after an interruption.
#
#   DATA=$HOME/data RUNS=$HOME/runs bash scripts/run_zero_shot.sh
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${DATA:-$HOME/data}"; RUNS="${RUNS:-$HOME/runs}"
mkdir -p "$RUNS/dets" "$RUNS/logs"
cd "$REPO"
nvidia-smi > "$RUNS/logs/nvidia-smi_$(date +%Y%m%d_%H%M%S).txt"

declare -A ROOT=([coco_val]="$DATA/coco/val2017" [kitti_val]="$DATA/kitti" [kitti_train]="$DATA/kitti"
                 [nuimages_val]="$DATA/nuimages" [nuimages_val_sub]="$DATA/nuimages")

run() {  # spec dataset [batch]
  local tag; tag=$(echo "$1" | sed 's#[:/@]#_#g')
  local out="$RUNS/dets/${tag}__$2.npz"
  [ -s "$out" ] && return
  echo "[$(date +%H:%M:%S)] $1 on $2"
  python -m detbench.infer --model "$1" --native "$DATA/native/$2.json" --images "${ROOT[$2]}" \
    --out "$out" --batch "${3:-8}" --workers 12 > "$RUNS/logs/${tag}__$2.log" 2>&1 \
    || echo "FAILED $1 on $2 (see $RUNS/logs/${tag}__$2.log)"
}

CLOSED=(tv:fasterrcnn_resnet50_fpn tv:retinanet_resnet50_fpn tv:fcos_resnet50_fpn
        tv:fasterrcnn_resnet50_fpn_v2 yolo:yolo11n.pt yolo:yolo11m.pt yolo:yolo11x.pt
        dfine:n dfine:m
        dfine:x)
OPEN=(gdino:IDEA-Research/grounding-dino-tiny
      gdino:openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_grit_v3det
      gdino:iSEE-Laboratory/llmdet_tiny)

# 1. main grid: every model on every dataset
for ds in kitti_val coco_val nuimages_val; do
  for m in "${CLOSED[@]}"; do run "$m" $ds; done
  for m in "${OPEN[@]}"; do run "$m" $ds 4; done
done
# 2. rider-fusion thresholds are fixed on KITTI train with one model
run tv:fasterrcnn_resnet50_fpn_v2 kitti_train
# 3. input-resolution and input-distortion ablations
for ds in kitti_val nuimages_val; do
  run yolo:yolo11m.pt@1280 $ds
  run dfine:m@letterbox $ds
done
# 4. prompt ablation on KITTI val and a log-spread fifth of nuImages val
for m in "${OPEN[@]}"; do
  for p in bare rider; do run "$m@$p" kitti_val 4; run "$m@$p" nuimages_val_sub 4; done
done
# 5. the Hugging Face port of D-FINE, for the implementation comparison
for m in n m x; do
  run "hfdfine:ustc-community/dfine-$( [ $m = n ] && echo nano || ([ $m = m ] && echo medium || echo xlarge) )-coco" coco_val
done
# 6. COCO 80-class sanity check for the open-vocabulary models
for m in "${OPEN[@]}"; do run "$m@coco80" coco_val 4; done
run yolo:yolo11n.pt@single coco_val   # Ultralytics prediction-mode (single-label) NMS, for the COCO check
echo "[$(date +%H:%M:%S)] zero-shot stage done"
