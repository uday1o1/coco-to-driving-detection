"""Turn stored native detections into scored detections for one protocol.

The switches mirror the ground-truth side (see taxonomy.py and data.coco_gt):

  semantics  label mapping, "corrected" or "tutorial"
  threshold  drop detections below the ngdet default score before computing
             mAP (0.3 closed set, 0.15 for Grounding DINO style models)
  minsize    drop detections shorter than the dataset's labelling minimum
  fusion     rider fusion (below) instead of mapping every bicycle to cyclist

Rider fusion. A COCO-trained detector sees a cyclist as two objects, a person
and a bicycle, while KITTI and nuImages label one box around both. Each person
is paired with at most one bicycle when the bicycle's score is at least
`bike_min`, the lower half of the person box overlaps the bicycle box
(intersection over the lower half >= `ioa`) and the person's horizontal centre
lies over the bicycle. Both thresholds are fixed on KITTI train (FUSION
below); without a score floor every pedestrian next to some faint bicycle
box would be absorbed, because detections are kept down to score 0.001. A pair becomes one cyclist
box (the union) scored by the geometric mean of the two scores; the person is
removed. Bicycles left without a rider are dropped, since parked bicycles are
not cyclists. Persons over a motorcycle are removed the same way, because both
datasets put the rider inside the motorcycle box. Pairs are taken greedily by
pair score. Fusion is applied only to datasets that put the rider inside the
bicycle box (taxonomy.RIDER_IN_BOX); COCO labels the rider as a person.
"""
from __future__ import annotations

import numpy as np

from . import taxonomy as tx

TUTORIAL_THRESHOLD = {"gdino": 0.15}
# Rider-fusion thresholds, chosen on KITTI train (scripts/analyze.py fusion).
FUSION = {"ioa": 0.7, "bike_min": 0.1}
TUTORIAL_THRESHOLD_DEFAULT = 0.3


def load(npz_path) -> dict:
    d = np.load(npz_path)
    return {k: d[k] for k in d.files}


def _pairs(person, other, ioa_min):
    """Candidate (person, other) index pairs and their IoA criterion."""
    if not len(person) or not len(other):
        return np.zeros((0, 2), int)
    p, o = person[:, None, :], other[None, :, :]
    mid = (p[..., 1] + p[..., 3]) / 2
    iw = np.clip(np.minimum(p[..., 2], o[..., 2]) - np.maximum(p[..., 0], o[..., 0]), 0, None)
    ih = np.clip(np.minimum(p[..., 3], o[..., 3]) - np.maximum(mid, o[..., 1]), 0, None)
    lower = (p[..., 2] - p[..., 0]) * (p[..., 3] - mid)
    ioa = iw * ih / np.maximum(lower, 1e-9)
    cx = (p[..., 0] + p[..., 2]) / 2
    ok = (ioa >= ioa_min) & (cx >= o[..., 0]) & (cx <= o[..., 2])
    return np.argwhere(ok)


def fuse_riders(boxes, scores, labels, ioa_min=0.7, bike_min=0.1, return_pairs=False):
    """Apply rider fusion to one image. labels are COCO-style names.
    Returns new (boxes, scores, labels), plus the (person, bicycle) index
    pairs when return_pairs is set."""
    person = np.nonzero(labels == "person")[0]
    out_b, out_s, out_l = [], [], []
    used_p, bike_pairs = set(), []
    for kind, new_label in (("bicycle", "cyclist"), ("motorcycle", None)):
        other = np.nonzero((labels == kind) & (scores >= bike_min))[0]
        cand = _pairs(boxes[person], boxes[other], ioa_min)
        if len(cand):
            ps = np.sqrt(scores[person[cand[:, 0]]] * scores[other[cand[:, 1]]])
            used_o = set()
            for k in np.argsort(-ps, kind="mergesort"):
                i, j = person[cand[k, 0]], other[cand[k, 1]]
                if i in used_p or j in used_o:
                    continue
                used_p.add(i); used_o.add(j)
                if new_label:
                    bike_pairs.append((i, j))
                    b = np.concatenate([np.minimum(boxes[i, :2], boxes[j, :2]),
                                        np.maximum(boxes[i, 2:], boxes[j, 2:])])
                    out_b.append(b); out_s.append(ps[k]); out_l.append(new_label)
    keep = [i for i in range(len(labels))
            if labels[i] not in ("bicycle", "motorcycle") and i not in used_p]
    b = np.concatenate([boxes[keep], np.array(out_b, np.float32).reshape(-1, 4)])
    s = np.concatenate([scores[keep], np.array(out_s, np.float32)])
    l = np.concatenate([labels[keep], np.array(out_l, dtype=labels.dtype)])
    return (b, s, l, bike_pairs) if return_pairs else (b, s, l)


def stretch_bicycles(boxes, scores, labels, factor=1.0):
    """Baseline for fusion: every bicycle becomes a cyclist box extended
    upwards by `factor` times its height, with no person pairing."""
    b, l = boxes.copy(), labels.copy()
    bike = l == "bicycle"
    b[bike, 1] -= factor * (b[bike, 3] - b[bike, 1])
    l[bike] = "cyclist"
    return b, scores, l


def apply(d: dict, dataset: str, semantics="corrected", threshold=False, minsize=True,
          fusion="rider", model_kind="tv", ioa=None, bike_min=None, stretch=1.0) -> dict:
    """Scored detections (image_id, boxes, scores, category_id) for one setting.
    fusion: "none" (bicycle -> cyclist), "rider" or "stretch"."""
    if not tx.RIDER_IN_BOX[dataset]:
        fusion = "none"
    names = np.array([tx.COCO_ALIASES.get(str(n).strip().lower(), str(n).strip().lower()) for n in d["names"]])
    labels = names[d["labels"]].astype("<U32")
    img, boxes, scores = d["image_id"], d["boxes"], d["scores"]
    if fusion != "none" and len(img):
        order = np.argsort(img, kind="mergesort")
        img, boxes, scores, labels = img[order], boxes[order], scores[order], labels[order]
        cuts = np.flatnonzero(np.diff(img)) + 1
        parts = []
        for sl in np.split(np.arange(len(img)), cuts):
            if not len(sl):
                continue
            f = fuse_riders if fusion == "rider" else stretch_bicycles
            kw = ({"ioa_min": FUSION["ioa"] if ioa is None else ioa,
                   "bike_min": FUSION["bike_min"] if bike_min is None else bike_min}
                  if fusion == "rider" else {"factor": stretch})
            b, s, l = f(boxes[sl], scores[sl], labels[sl], **kw)
            parts.append((np.full(len(s), img[sl[0]]), b, s, l))
        img, boxes, scores, labels = (np.concatenate(x) for x in zip(*parts))
    cls = np.array([tx.pred_class(l, semantics) or "" for l in labels])
    keep = cls != ""
    if threshold:
        keep &= scores >= TUTORIAL_THRESHOLD.get(model_kind, TUTORIAL_THRESHOLD_DEFAULT)
    if minsize:
        keep &= (boxes[:, 3] - boxes[:, 1]) >= tx.MIN_HEIGHT[dataset]
    cid = np.array([tx.CID.get(c, -1) + 1 for c in cls])
    return {"image_id": img[keep], "boxes": boxes[keep].astype(np.float64),
            "scores": scores[keep].astype(np.float64), "category_id": cid[keep]}
