"""Ground truth for COCO val2017, KITTI and nuImages.

`build` reads a dataset once into a native record (every box with its
dataset-native class), saved as JSON. `coco_gt` turns a native record into a
COCO-format ground truth for one protocol setting, so the protocol audit can
vary the setting without re-reading the dataset.

Ignore regions are iscrowd=1 boxes. pycocotools never counts a crowd box as a
miss, and a detection covering one (intersection over the detection's own
area at or above the IoU threshold) is neither a true nor a false positive.
A region ignored for every class is written once per class. Box area (w*h) is
used on all datasets so size bins mean the same thing everywhere.

Extra fields kept for the subgroup analysis:
  image: cluster (resampling unit), city, camera, sun_elev, lighting
  box:   fine (native class), height, occluded, truncated, difficulty
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image

from . import taxonomy as tx
from .solar import CITY_LATLON, lighting, solar_elevation

CATEGORIES = [{"id": i + 1, "name": c} for i, c in enumerate(tx.CLASSES)]


def kitti_difficulty(h, occ, trunc):
    """KITTI benchmark difficulty: 0 easy, 1 moderate, 2 hard, 3 beyond hard."""
    for level, (min_h, max_occ, max_tr) in enumerate(
            [(40, 0, 0.15), (25, 1, 0.30), (25, 2, 0.50)]):
        if h >= min_h and occ <= max_occ and trunc <= max_tr:
            return level
    return 3


def kitti_drives(root: Path) -> dict[int, str]:
    """Image index -> raw drive name, from the object devkit mapping files."""
    rand = [int(v) for v in (root / "devkit/mapping/train_rand.txt").read_text().split(",")]
    lines = (root / "devkit/mapping/train_mapping.txt").read_text().splitlines()
    return {i: lines[r - 1].split()[1] for i, r in enumerate(rand)}


def build_kitti(root: Path, split: str):
    images, boxes = [], []
    drives = kitti_drives(root)
    for stem in (root / f"ImageSets/{split}.txt").read_text().split():
        iid = int(stem)
        img = root / f"training/image_2/{stem}.png"
        w, h = Image.open(img).size if img.exists() else (None, None)
        images.append(dict(id=iid, file_name=f"training/image_2/{stem}.png", width=w,
                           height=h, cluster=drives[iid], camera="front"))
        for line in (root / f"training/label_2/{stem}.txt").read_text().splitlines():
            f = line.split()
            trunc, occ = float(f[1]), int(f[2])
            x1, y1, x2, y2 = map(float, f[4:8])
            boxes.append(dict(image_id=iid, xyxy=[x1, y1, x2, y2], fine=f[0], crowd=False,
                              occluded=occ, truncated=trunc,
                              difficulty=kitti_difficulty(y2 - y1, occ, trunc)))
    return images, boxes


def build_nuimages(root: Path, version: str):
    load = lambda n: json.loads((root / version / f"{n}.json").read_text())
    cat = {c["token"]: c["name"] for c in load("category")}
    attr = {a["token"]: a["name"] for a in load("attribute")}
    logs = {lg["token"]: lg for lg in load("log")}
    samples = {s["token"]: s for s in load("sample")}
    keyframes = {s["token"]: s for s in load("sample_data") if s["is_key_frame"]}
    by_image = defaultdict(list)
    for o in load("object_ann"):
        if o["sample_data_token"] in keyframes:
            by_image[o["sample_data_token"]].append(o)
    images, boxes = [], []
    for iid, tok in enumerate(sorted(keyframes)):
        sd = keyframes[tok]
        log = logs[samples[sd["sample_token"]]["log_token"]]
        elev = solar_elevation(sd["timestamp"] / 1e6, *CITY_LATLON[log["location"]])
        images.append(dict(id=iid, file_name=sd["filename"], width=sd["width"],
                           height=sd["height"], cluster=log["logfile"],
                           city=log["location"].split("-")[0],
                           camera=sd["filename"].split("/")[1], sun_elev=round(elev, 2),
                           lighting=lighting(elev), token=tok))
        for o in by_image.get(tok, []):
            name = cat[o["category_token"]]
            if name == "vehicle.bicycle":
                names = {attr[a] for a in o["attribute_tokens"]}
                if "cycle.with_rider" in names:
                    name += ".with_rider"
                elif "cycle.without_rider" in names:
                    name += ".without_rider"
            boxes.append(dict(image_id=iid, xyxy=[float(v) for v in o["bbox"]],
                              fine=name, crowd=False))
    return images, boxes


def build_coco(ann_file: Path):
    d = json.loads(ann_file.read_text())
    names = {c["id"]: c["name"] for c in d["categories"]}
    images = [dict(id=im["id"], file_name=im["file_name"], width=im["width"],
                   height=im["height"], cluster=str(im["id"]), camera="mixed")
              for im in d["images"]]
    boxes = []
    for a in d["annotations"]:
        x, y, w, h = a["bbox"]
        boxes.append(dict(image_id=a["image_id"], xyxy=[x, y, x + w, y + h],
                          fine=names[a["category_id"]], crowd=bool(a["iscrowd"])))
    return images, boxes


def coco_gt(native: dict, semantics="corrected", ignore=True, minsize=True, empty=True):
    """COCO-format ground truth for one protocol setting.

    empty=False drops nuImages frames with no native boxes, as ngdet's
    nuImages loader does (its COCO and KITTI loaders keep every image).
    """
    ds = native["dataset"]
    min_h = tx.MIN_HEIGHT[ds] if minsize else 0.0
    has_box = {b["image_id"] for b in native["boxes"]}
    images = [im for im in native["images"] if empty or ds != "nuimages" or im["id"] in has_box]
    keep = {im["id"] for im in images}
    anns = []
    for i, b in enumerate(native["boxes"]):
        if b["image_id"] not in keep:
            continue
        cls, scope = tx.gt_rule(ds, b["fine"], b["crowd"], semantics, ignore)
        x1, y1, x2, y2 = b["xyxy"]
        w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
        if cls is not None and scope is None and h < min_h:
            scope = "class"
        if scope == "all":
            targets, crowd = tx.CLASSES, 1
        elif cls is None:
            continue
        else:
            targets, crowd = (cls,), int(scope == "class")
        for c in targets:
            anns.append({"id": len(anns) + 1, "image_id": b["image_id"],
                         "category_id": tx.CID[c] + 1, "bbox": [x1, y1, w, h],
                         "area": w * h, "iscrowd": crowd, "height": h,
                         "native_index": i})
    return {"images": images, "annotations": anns, "categories": CATEGORIES}


def main():
    p = argparse.ArgumentParser(description="Read a dataset into a native ground-truth record.")
    p.add_argument("dataset", choices=["coco", "kitti", "nuimages"])
    p.add_argument("--root", type=Path, required=True,
                   help="kitti: object dir with training/ ImageSets/ devkit/; "
                        "nuimages: dir with v1.0-<split>/; coco: instances_val2017.json")
    p.add_argument("--split", default="val")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.dataset == "kitti":
        images, boxes = build_kitti(a.root, a.split)
    elif a.dataset == "nuimages":
        images, boxes = build_nuimages(a.root, f"v1.0-{a.split}")
    else:
        images, boxes = build_coco(a.root)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"dataset": a.dataset, "split": a.split,
                                 "images": images, "boxes": boxes}))
    print(f"{a.out}: {len(images)} images, {len(boxes)} native boxes")


if __name__ == "__main__":
    main()
