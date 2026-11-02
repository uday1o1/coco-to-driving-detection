"""Draw the failure examples used in the report (figures/qualitative.jpg).

Four panels, picked automatically so the selection is reproducible:
  a  KITTI: a cyclist seen as a person plus a bicycle (raw output)
  b  the same image after rider fusion
  c  KITTI: confident detections inside DontCare regions
  d  nuImages: missed construction workers (daytime frame) at the person
     operating threshold, cropped around them

    python scripts/qualitative.py --runs $RUNS --data $DATA --model tv:fasterrcnn_resnet50_fpn_v2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from detbench import analysis as A, protocol  # noqa: E402
from detbench.evaluate import _iou, match  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
COLORS = {"person": (230, 60, 60), "bicycle": (255, 170, 0), "cyclist": (170, 60, 230),
          "car": (60, 120, 230), "vehicle": (60, 120, 230), "truck": (60, 120, 230), "bus": (60, 120, 230)}


def boxes_for(d, iid, thr, names_keep):
    names = np.array([str(n).lower() for n in d["names"]])
    sel = (d["image_id"] == iid) & (d["scores"] >= thr)
    lab = names[d["labels"][sel]]
    k = np.isin(lab, names_keep)
    return d["boxes"][sel][k], d["scores"][sel][k], lab[k]


def draw(img, gt_boxes, preds, ignore=(), title=""):
    im = img.convert("RGB").copy()
    dr = ImageDraw.Draw(im)
    for b in ignore:
        dr.rectangle(list(b), outline=(150, 150, 150), width=2)
    for b in gt_boxes:
        dr.rectangle(list(b), outline=(40, 200, 40), width=3)
    for b, s, l in zip(*preds):
        dr.rectangle(list(map(float, b)), outline=COLORS.get(l, (255, 255, 255)), width=2)
        dr.text((float(b[0]) + 2, float(b[1]) - 11), f"{l} {s:.2f}", fill=COLORS.get(l, (255, 255, 255)))
    dr.rectangle([0, 0, im.width, 16], fill=(0, 0, 0))
    dr.text((4, 2), title, fill=(255, 255, 255))
    return im


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--model", default="tv:fasterrcnn_resnet50_fpn_v2")
    p.add_argument("--list-images", action="store_true", help="print the image files needed and exit")
    a = p.parse_args()
    tag = a.model.replace(":", "_").replace("/", "_")
    kitti = A.native(str(a.data / "native/kitti_val.json"))
    dk = protocol.load(a.runs / f"dets/{tag}__kitti_val.npz")
    by = {}
    for b in kitti["boxes"]:
        by.setdefault(b["image_id"], []).append(b)
    thr = 0.5
    # a: image with the most cyclists that the raw output covers with person + bicycle
    best, best_n = None, 0
    for iid, bs in by.items():
        cyc = [b["xyxy"] for b in bs if b["fine"] == "Cyclist" and b["xyxy"][3] - b["xyxy"][1] > 60]
        if not cyc:
            continue
        pb, ps, pl = boxes_for(dk, iid, thr, ["person", "bicycle"])
        n = sum(1 for c in cyc if len(pb) and (_iou(pb.astype(float), np.array([c], float),
                                                     np.zeros(1, bool)) > 0.3).any())
        if n > best_n:
            best, best_n = iid, n
    file_of = {im["id"]: im["file_name"] for im in kitti["images"]}
    # c: most confident detections falling inside DontCare regions
    best_c, best_v = None, 0
    for iid, bs in by.items():
        dc = np.array([b["xyxy"] for b in bs if b["fine"] == "DontCare"], float).reshape(-1, 4)
        if not len(dc):
            continue
        pb, ps, pl = boxes_for(dk, iid, thr, ["car", "truck", "person"])
        if not len(pb):
            continue
        inside = (_iou(pb.astype(float), dc, np.ones(len(dc), bool)) >= 0.5).any(1)
        v = float(ps[inside].sum())
        if v > best_v:
            best_c, best_v = iid, v
    # d: daytime nuImages frame with the most missed construction workers
    nat, g, dets = A.prepare(a.runs / f"dets/{tag}__nuimages_val.npz", a.data / "native/nuimages_val.json",
                             a.model, A.settings())
    m, ids = match(g, dets, cat_ids=[2])
    pm = m[2]
    t, _ = A.operating_threshold(pm)
    anns = g["annotations"]
    rec = {im["id"]: im for im in nat["images"]}
    miss = {}
    for k, sc, ig in zip(pm.gt_ann, pm.gt_score50, pm.gt_ignored):
        b = nat["boxes"][anns[k]["native_index"]]
        if (not ig and sc < t and b["fine"] == "human.pedestrian.construction_worker"
                and rec[b["image_id"]]["lighting"] == "day"):
            miss[b["image_id"]] = miss.get(b["image_id"], 0) + 1
    iid = max(miss, key=miss.get)
    im_rec = rec[iid]
    if a.list_images:
        print(f"kitti/{file_of[best]}\nkitti/{file_of[best_c]}\nnuimages/{im_rec['file_name']}")
        return

    img = Image.open(a.data / "kitti" / file_of[best])
    gt = [b["xyxy"] for b in by[best] if b["fine"] in ("Cyclist", "Pedestrian")]
    raw = boxes_for(dk, best, thr, ["person", "bicycle"])
    fb, fs, fl = protocol.fuse_riders(raw[0], raw[1], raw[2], protocol.FUSION["ioa"], protocol.FUSION["bike_min"])
    panel_a = draw(img, gt, raw, title="(a) KITTI, raw COCO labels: rider and bicycle are separate boxes")
    panel_b = draw(img, gt, (fb, fs, fl), title="(b) after rider fusion: one cyclist box")
    img = Image.open(a.data / "kitti" / file_of[best_c])
    dc = [b["xyxy"] for b in by[best_c] if b["fine"] == "DontCare"]
    gt = [b["xyxy"] for b in by[best_c] if b["fine"] in ("Car", "Van", "Truck", "Pedestrian", "Cyclist")]
    panel_c = draw(img, gt, boxes_for(dk, best_c, thr, ["car", "truck", "person"]), dc,
                   title="(c) KITTI: detections in unlabeled DontCare regions (grey)")
    img = Image.open(a.data / "nuimages" / im_rec["file_name"])
    peds = [b for b in nat["boxes"] if b["image_id"] == iid and b["fine"].startswith("human.pedestrian")]
    workers = np.array([b["xyxy"] for b in peds if b["fine"].endswith("construction_worker")])
    others = [b["xyxy"] for b in peds if not b["fine"].endswith("construction_worker")]
    pad = 60
    x1 = int(max(0, workers[:, 0].min() - pad)); x2 = int(min(img.width, workers[:, 2].max() + pad))
    cy = workers[:, [1, 3]].mean(); half = max(90, (workers[:, 3].max() - workers[:, 1].min()) / 2 + pad)
    y1 = int(max(0, cy - half)); y2 = int(min(img.height, cy + half))
    off = np.array([x1, y1, x1, y1], float)
    pb, ps, pl = boxes_for(protocol.load(a.runs / f"dets/{tag}__nuimages_val.npz"), iid, t, ["person"])
    inside = (pb[:, 2] > x1) & (pb[:, 0] < x2) & (pb[:, 3] > y1) & (pb[:, 1] < y2)
    shift = lambda bs: [list(np.array(b) - off) for b in bs]
    panel_d = draw(img.crop((x1, y1, x2, y2)), shift(workers), (pb[inside] - off, ps[inside], pl[inside]),
                   ignore=shift(others),
                   title=f"(d) nuImages, crop: construction workers (green), other pedestrians (grey), "
                         f"person detections at the F1 threshold {t:.2f} (red)")
    W = 1600
    rows = [p.resize((W, int(p.height * W / p.width))) for p in (panel_a, panel_b, panel_c, panel_d)]
    canvas = Image.new("RGB", (W, sum(r.height for r in rows) + 6 * 3), (255, 255, 255))
    y = 0
    for r in rows:
        canvas.paste(r, (0, y)); y += r.height + 6
    out = REPO / "figures/qualitative.jpg"
    out.parent.mkdir(exist_ok=True)
    canvas.save(out, quality=88)
    print(f"wrote {out}: kitti {best}, {best_c}; nuimages {im_rec['file_name']}")


if __name__ == "__main__":
    main()
