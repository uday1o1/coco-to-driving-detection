#!/usr/bin/env bash
# Fine-tune YOLO11-M and Faster R-CNN v2 on KITTI train with three seeds each,
# then run every fine-tuned model on the three evaluation sets.
#
#   DATA=$HOME/data RUNS=$HOME/runs bash scripts/run_finetune.sh
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${DATA:-$HOME/data}"; RUNS="${RUNS:-$HOME/runs}"
mkdir -p "$RUNS/ft" "$RUNS/dets" "$RUNS/logs"
cd "$REPO"
nvidia-smi > "$RUNS/logs/nvidia-smi_ft_$(date +%Y%m%d_%H%M%S).txt"

for seed in 0 1 2; do
  [ -s "$RUNS/ft/yolo11m_seed$seed/weights/last.pt" ] || \
    python -m detbench.finetune yolo --kitti "$DATA/kitti" --native "$DATA/native/kitti_train.json" \
      --seed $seed --out "$RUNS/ft" > "$RUNS/logs/ft_yolo11m_seed$seed.log" 2>&1 || echo "FAILED yolo seed $seed"
  [ -s "$RUNS/ft/frcnn_v2_seed$seed/last.pt" ] || \
    python -m detbench.finetune frcnn --kitti "$DATA/kitti" --native "$DATA/native/kitti_train.json" \
      --seed $seed --out "$RUNS/ft" > "$RUNS/logs/ft_frcnn_v2_seed$seed.log" 2>&1 || echo "FAILED frcnn seed $seed"
done

declare -A ROOT=([coco_val]="$DATA/coco/val2017" [kitti_val]="$DATA/kitti" [nuimages_val]="$DATA/nuimages")
for seed in 0 1 2; do
  for spec in "yolo:ft_yolo11m_seed$seed#$RUNS/ft/yolo11m_seed$seed/weights/last.pt" \
              "tv:fasterrcnn_resnet50_fpn_v2#$RUNS/ft/frcnn_v2_seed$seed/last.pt"; do
    tag=$( [[ $spec == yolo* ]] && echo "ft_yolo11m_seed$seed" || echo "ft_frcnn_v2_seed$seed" )
    for ds in kitti_val nuimages_val coco_val; do
      out="$RUNS/dets/${tag}__$ds.npz"
      [ -s "$out" ] && continue
      python -m detbench.infer --model "$spec" --native "$DATA/native/$ds.json" --images "${ROOT[$ds]}" \
        --out "$out" --batch 8 --workers 12 > "$RUNS/logs/${tag}__$ds.log" 2>&1 || echo "FAILED $tag on $ds"
    done
  done
done
echo "[$(date +%H:%M:%S)] fine-tuning stage done"
