#!/usr/bin/env bash
# Install dependencies and download the three datasets.
#
#   DATA=/path/to/data bash scripts/setup.sh
#
# Needs about 45 GB free during extraction (30 GB after). Every download is
# public and needs no account: COCO from cocodataset.org, KITTI object from
# the KITTI S3 bucket, nuImages from the nuScenes CloudFront bucket. Only the
# nuImages validation images are extracted. Re-running skips finished steps.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${DATA:-$HOME/data}"
NO_IMAGES="${NO_IMAGES:-0}"   # 1: annotations and metadata only, enough to re-run the analysis
mkdir -p "$DATA" && cd "$DATA"

pip install -q -r "$REPO/requirements.txt"

# official D-FINE implementation (weights are fetched on first use)
DFINE_SRC="${DFINE_SRC:-$HOME/dfine_src}"
if [ ! -d "$DFINE_SRC" ]; then
  git clone -q https://github.com/Peterande/D-FINE.git "$DFINE_SRC"
  git -C "$DFINE_SRC" checkout -q 956d1709314c2c6a4df6f34de232054578a7449f
fi

fetch() {  # url file
  [ -s "$2" ] && return
  if command -v aria2c >/dev/null; then aria2c -q -x 16 -s 16 -o "$2" "$1"
  else curl -fsSL --retry 5 -o "$2" "$1"; fi
}

# COCO val2017 images and annotations
if [ "$NO_IMAGES" = 1 ] && [ ! -f coco/annotations/instances_val2017.json ]; then
  fetch http://images.cocodataset.org/annotations/annotations_trainval2017.zip coco_ann.zip
  mkdir -p coco && unzip -q -o coco_ann.zip -d coco && rm coco_ann.zip
elif [ "$NO_IMAGES" != 1 ] && [ ! -d coco/val2017 ]; then
  fetch http://images.cocodataset.org/zips/val2017.zip val2017.zip
  fetch http://images.cocodataset.org/annotations/annotations_trainval2017.zip coco_ann.zip
  mkdir -p coco && unzip -q val2017.zip -d coco && unzip -q -o coco_ann.zip -d coco
  rm val2017.zip coco_ann.zip
fi

# KITTI object: left colour images, labels, devkit (drive mapping), Chen et al. split
K=https://s3.eu-central-1.amazonaws.com/avg-kitti
if [ "$NO_IMAGES" = 1 ] && [ ! -d kitti/training/label_2 ]; then
  fetch $K/data_object_label_2.zip kitti_lbl.zip
  fetch $K/devkit_object.zip kitti_devkit.zip
  mkdir -p kitti && unzip -q -o kitti_lbl.zip -d kitti && unzip -q -o kitti_devkit.zip -d kitti/devkit
  rm kitti_lbl.zip kitti_devkit.zip
elif [ "$NO_IMAGES" != 1 ] && [ ! -d kitti/training/image_2 ]; then
  fetch $K/data_object_image_2.zip kitti_img.zip
  fetch $K/data_object_label_2.zip kitti_lbl.zip
  fetch $K/devkit_object.zip kitti_devkit.zip
  mkdir -p kitti && unzip -q kitti_img.zip 'training/*' -d kitti && unzip -q -o kitti_lbl.zip -d kitti
  unzip -q -o kitti_devkit.zip -d kitti/devkit
  rm kitti_img.zip kitti_lbl.zip kitti_devkit.zip
fi
mkdir -p kitti/ImageSets && cp "$REPO"/data/kitti_split/*.txt kitti/ImageSets/

# nuImages: metadata for all splits, validation key frames only
N=https://d36yt3mvayqw5m.cloudfront.net/public/nuimages-v1.0
if [ ! -d nuimages/v1.0-val ] || { [ "$NO_IMAGES" != 1 ] && [ ! -d nuimages/samples ]; }; then
  fetch $N/nuimages-v1.0-all-metadata.tgz nuim_meta.tgz
  mkdir -p nuimages && tar xzf nuim_meta.tgz -C nuimages
  python - <<'EOF' > nuim_val_files.txt
import json
for s in json.load(open("nuimages/v1.0-val/sample_data.json")):
    if s["is_key_frame"]:
        print(s["filename"])
EOF
  if [ "$NO_IMAGES" != 1 ]; then
    fetch $N/nuimages-v1.0-all-samples.tgz nuim_samples.tgz
    tar xzf nuim_samples.tgz -C nuimages -T nuim_val_files.txt
    rm nuim_samples.tgz
  fi
  rm nuim_meta.tgz
fi

# Native ground-truth records
cd "$REPO"
mkdir -p "$DATA/native"
python -m detbench.data coco --root "$DATA/coco/annotations/instances_val2017.json" --out "$DATA/native/coco_val.json"
for s in train val; do
  python -m detbench.data kitti --root "$DATA/kitti" --split $s --out "$DATA/native/kitti_$s.json"
done
python -m detbench.data nuimages --root "$DATA/nuimages" --out "$DATA/native/nuimages_val.json"
# the tutorial's evaluation subsets: KITTI first 120 of ngdet's own seed-42 split and of train.txt,
# COCO first 120 val images, first 500 annotated nuImages val frames
for s in ngdet120 train120; do
  python -m detbench.data kitti --root "$DATA/kitti" --split $s --out "$DATA/native/kitti_$s.json"
done
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
# every fifth nuImages val frame (spread over all logs), for the prompt ablation
python - "$DATA/native" <<'EOF'
import json, sys
d = json.load(open(f"{sys.argv[1]}/nuimages_val.json"))
keep = {im["id"] for im in d["images"][::5]}
d["images"] = [im for im in d["images"] if im["id"] in keep]
d["boxes"] = [b for b in d["boxes"] if b["image_id"] in keep]
json.dump(d, open(f"{sys.argv[1]}/nuimages_val_sub.json", "w"))
EOF
echo "setup done: $(du -sh "$DATA" | cut -f1) in $DATA"
