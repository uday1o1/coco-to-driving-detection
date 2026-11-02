#!/usr/bin/env bash
# Reproduce the ngdet tutorial's zero-shot table (TUTORIAL.md section 12.1)
# with ngdet's own code at a pinned commit, on the same images it used:
#   KITTI: first 120 images of ngdet's own seed-42 80/20 "train" split
#   COCO:  first 120 val2017 images (sorted ids)
#   nuImages: first 500 annotated v1.0-val key frames (sorted tokens)
# then run this repo's wrappers on the same images, one image per batch like
# ngdet, so the two harnesses can be compared image for image.
#
#   DATA=$HOME/data RUNS=$HOME/runs bash scripts/reproduce_tutorial.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${DATA:-$HOME/data}"; RUNS="${RUNS:-$HOME/runs}"
NGDET_COMMIT=e347f6ff4468cc33442f38277119bb7bdc18a8d1
NG="$RUNS/ngdet_src"; OUT="$RUNS/ngdet_out"
mkdir -p "$OUT" "$RUNS/dets" "$RUNS/logs"

if [ ! -d "$NG" ]; then
  git clone -q https://github.com/lkk688/DeepDataMiningLearning.git "$NG"
  git -C "$NG" checkout -q $NGDET_COMMIT
fi
pip install -q opencv-python-headless pyarrow

# The tutorial's KITTI numbers come from the first 120 images of ImageSets/train.txt
# (the Chen et al. split used here); without ImageSets/ ngdet builds its own
# seed-42 80/20 split instead. Both are run.
KN="$DATA/kitti_ngdet"
mkdir -p "$KN" && ln -sfn "$DATA/kitti/training" "$KN/training"

MODELS=(torchvision:fasterrcnn_resnet50_fpn_v2 yolo:yolo11x.pt gdino:IDEA-Research/grounding-dino-tiny@0.15)
cd "$NG"
[ -s "$OUT/k_c/metrics.json" ] || python -m DeepDataMiningLearning.ngdet.run_eval --models "${MODELS[@]}" --datasets kitti coco \
  --max-images 120 --score-thr 0.3 --device cuda --out-dir "$OUT/k_c" \
  --coco-ann "$DATA/coco/annotations/instances_val2017.json" \
  --roots kitti="$KN" coco="$DATA/coco/val2017" > "$RUNS/logs/ngdet_kitti_coco.log" 2>&1
[ -s "$OUT/nuim/metrics.json" ] || python -m DeepDataMiningLearning.ngdet.run_eval --models "${MODELS[@]}" --datasets nuimages \
  --nuimages-version v1.0-val --max-images 500 --score-thr 0.3 --device cuda --out-dir "$OUT/nuim" \
  --roots nuimages="$DATA/nuimages" > "$RUNS/logs/ngdet_nuimages.log" 2>&1
[ -s "$OUT/k_train120/metrics.json" ] || python -m DeepDataMiningLearning.ngdet.run_eval --models "${MODELS[@]}" --datasets kitti \
  --max-images 120 --score-thr 0.3 --device cuda --out-dir "$OUT/k_train120" \
  --roots kitti="$DATA/kitti" > "$RUNS/logs/ngdet_kitti_train120.log" 2>&1

# the same images as native records for this repo
cd "$REPO"
head -n 120 "$KN/ImageSets/train.txt" > "$DATA/kitti/ImageSets/ngdet120.txt"
python -m detbench.data kitti --root "$DATA/kitti" --split ngdet120 --out "$DATA/native/kitti_ngdet120.json"
head -n 120 "$DATA/kitti/ImageSets/train.txt" > "$DATA/kitti/ImageSets/train120.txt"
python -m detbench.data kitti --root "$DATA/kitti" --split train120 --out "$DATA/native/kitti_train120.json"
python - "$DATA/native" <<'EOF'
import json, sys
d = sys.argv[1]
c = json.load(open(f"{d}/coco_val.json"))
keep = set(sorted(im["id"] for im in c["images"])[:120])
json.dump({**c, "images": [i for i in c["images"] if i["id"] in keep],
           "boxes": [b for b in c["boxes"] if b["image_id"] in keep]}, open(f"{d}/coco_first120.json", "w"))
n = json.load(open(f"{d}/nuimages_val.json"))
annotated = {b["image_id"] for b in n["boxes"]}
keep = {im["id"] for im in [i for i in n["images"] if i["id"] in annotated][:500]}
json.dump({**n, "images": [i for i in n["images"] if i["id"] in keep],
           "boxes": [b for b in n["boxes"] if b["image_id"] in keep]}, open(f"{d}/nuimages_ngdet500.json", "w"))
EOF
declare -A ROOT=([kitti_ngdet120]="$DATA/kitti" [kitti_train120]="$DATA/kitti" [coco_first120]="$DATA/coco/val2017" [nuimages_ngdet500]="$DATA/nuimages")
for ds in kitti_ngdet120 kitti_train120 coco_first120 nuimages_ngdet500; do
  for m in tv:fasterrcnn_resnet50_fpn_v2 yolo:yolo11x.pt@single gdino:IDEA-Research/grounding-dino-tiny; do
    tag=$(echo "$m" | sed 's#[:/@]#_#g')
    [ -s "$RUNS/dets/${tag}__$ds.npz" ] || python -m detbench.infer --model "$m" \
      --native "$DATA/native/$ds.json" --images "${ROOT[$ds]}" --out "$RUNS/dets/${tag}__$ds.npz" \
      --batch 1 --workers 8 > "$RUNS/logs/${tag}__$ds.log" 2>&1
  done
done
echo "tutorial reproduction done"
