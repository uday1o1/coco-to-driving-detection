"""Official 80-class COCO val2017 AP for a stored detection file.

A wrapper passes only if this reproduces the checkpoint's published COCO AP,
which catches preprocessing, box-format and label-mapping bugs before any
cross-dataset number is trusted. Uses the official annotation file and the
standard pycocotools protocol (mask-area size buckets).

    python -m detbench.coco_gate runs/dets/yolo11m__coco_val.npz \
        --ann $DATA/coco/annotations/instances_val2017.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import numpy as np


def coco_ap(npz: Path, ann: Path, image_ids=None) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    d = np.load(npz)
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ann))
    name_to_cat = {c["name"]: c["id"] for c in gt.dataset["categories"]}
    cats = np.array([name_to_cat.get(str(n), -1) for n in d["names"]])
    c = cats[d["labels"]]
    keep = c > 0
    b = d["boxes"][keep]
    rows = np.column_stack([d["image_id"][keep], b[:, 0], b[:, 1], b[:, 2] - b[:, 0],
                            b[:, 3] - b[:, 1], d["scores"][keep], c[keep]])
    with contextlib.redirect_stdout(io.StringIO()):
        E = COCOeval(gt, gt.loadRes(rows), "bbox")
        if image_ids is not None:
            E.params.imgIds = list(image_ids)
        E.evaluate(); E.accumulate(); E.summarize()
    keys = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]
    return dict(zip(keys, (round(float(v), 4) for v in E.stats)))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("npz", type=Path, nargs="+")
    p.add_argument("--ann", type=Path, required=True)
    a = p.parse_args()
    for f in a.npz:
        print(f.stem, json.dumps(coco_ap(f, a.ann)))


if __name__ == "__main__":
    main()
