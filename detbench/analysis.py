"""Metrics for one (detections, dataset, protocol) triple.

Every number goes through two independent implementations: faster-coco-eval
for the canonical 12-value COCO table, and this repo's matcher for per-class
AP, height bins, bootstrap intervals and per-object recall. `summarize`
asserts the two agree on AP@[.5:.95] before returning.
"""
from __future__ import annotations

import contextlib
import io
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import protocol, taxonomy as tx
from .data import coco_gt
from .evaluate import APCache, ci, cluster_weights, match, mean_ap

HEIGHT_BINS = [(0, 40), (40, 80), (80, 160), (160, 1e10)]
STAT_KEYS = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]


@lru_cache(maxsize=16)
def native(path: str) -> dict:
    return json.loads(Path(path).read_text())


def model_kind(spec: str) -> str:
    return spec.split(":")[0].split("_")[0]


def coco_stats(gt: dict, dets: dict) -> dict:
    from faster_coco_eval import COCO, COCOeval_faster
    if not len(dets["scores"]):
        return {k: 0.0 for k in STAT_KEYS}
    with contextlib.redirect_stdout(io.StringIO()):
        g = COCO(gt)
        b = dets["boxes"]
        rows = np.column_stack([dets["image_id"], b[:, 0], b[:, 1], b[:, 2] - b[:, 0],
                                b[:, 3] - b[:, 1], dets["scores"], dets["category_id"]])
        E = COCOeval_faster(g, g.loadRes(rows), "bbox")
        E.evaluate(); E.accumulate(); E.summarize()
    return dict(zip(STAT_KEYS, (float(v) for v in E.stats[:12])))


def settings(**over):
    """Protocol switches; defaults are the corrected protocol."""
    s = dict(semantics="corrected", ignore=True, minsize=True, empty=True, threshold=False, fusion="rider")
    s.update(over)
    return s


def prepare(det_path, native_path, spec, s, ioa=0.3, stretch=1.0):
    nat = native(str(native_path))
    gt = coco_gt(nat, s["semantics"], s["ignore"], s["minsize"], s["empty"])
    dets = protocol.apply(protocol.load(det_path), nat["dataset"], s["semantics"], s["threshold"],
                          s["minsize"], s["fusion"], model_kind(spec), ioa, stretch)
    keep = np.isin(dets["image_id"], [im["id"] for im in gt["images"]])
    dets = {k: v[keep] for k, v in dets.items()}
    return nat, gt, dets


def summarize(det_path, native_path, spec, s=None, B=1000, heights=True, keep_match=False):
    s = s or settings()
    nat, gt, dets = prepare(det_path, native_path, spec, s)
    stats = coco_stats(gt, dets)
    m, image_ids = match(gt, dets)
    caches = {c: APCache(m[c]) for c in m}
    ap = mean_ap(caches.values())
    assert abs(ap - stats["AP"]) < 1e-6, f"matcher {ap} vs faster-coco-eval {stats['AP']}"
    out = {"stats": stats, "per_class": {}, "n_images": len(gt["images"]),
           "n_gt": {tx.CLASSES[c - 1]: int(m[c].npig.sum()) for c in m}}
    for c, cache in caches.items():
        out["per_class"][tx.CLASSES[c - 1]] = {"AP": cache.ap(), "AP50": cache.ap(thr_index=0),
                                               "AP75": cache.ap(thr_index=5)}
    if heights:
        out["height_bins"] = {}
        for lo, hi in HEIGHT_BINS:
            mb, _ = match(gt, dets, "height", lo, hi)
            out["height_bins"][f"{lo}-{int(hi) if hi < 1e9 else 'inf'}"] = {
                tx.CLASSES[c - 1]: APCache(mb[c]).ap() for c in mb}
    if B:
        clusters = np.array([{im["id"]: im["cluster"] for im in nat["images"]}[i] for i in image_ids])
        boot = {"AP": [], "AP50": [], **{f"AP_{tx.CLASSES[c - 1]}": [] for c in caches}}
        for w in cluster_weights(clusters, B):
            boot["AP"].append(mean_ap(caches.values(), w))
            boot["AP50"].append(mean_ap(caches.values(), w, 0))
            for c, cache in caches.items():
                v = cache.ap(w)
                boot[f"AP_{tx.CLASSES[c - 1]}"].append(v if v >= 0 else np.nan)
        out["ci95"] = {k: ci(v) for k, v in boot.items()}
        out["n_clusters"] = int(len(np.unique(clusters)))
    if keep_match:
        out["_match"], out["_image_ids"], out["_gt"], out["_native"] = m, image_ids, gt, nat
    return out


def paired_diff(det_a, det_b, native_path, spec_a, spec_b, s_a=None, s_b=None, B=1000, cls=None):
    """Clustered paired bootstrap of AP(a) - AP(b) on the same images."""
    s_a, s_b = s_a or settings(), s_b or settings()
    res = []
    for det, spec, s in ((det_a, spec_a, s_a), (det_b, spec_b, s_b)):
        nat, gt, dets = prepare(det, native_path, spec, s)
        m, ids = match(gt, dets)
        res.append(({c: APCache(m[c]) for c in m}, ids, nat))
    (ca, ida, nat), (cb, idb, _) = res
    assert np.array_equal(ida, idb), "paired comparison needs the same image set"
    pick = (lambda c: [c[cls]]) if cls else (lambda c: list(c.values()))
    clusters = np.array([{im["id"]: im["cluster"] for im in nat["images"]}[i] for i in ida])
    d0 = mean_ap(pick(ca)) - mean_ap(pick(cb))
    boot = [mean_ap(pick(ca), w) - mean_ap(pick(cb), w) for w in cluster_weights(clusters, B, seed=1)]
    return {"diff": d0, "ci95": ci(boot), "p_le_0": float(np.mean(np.array(boot) <= 0))}


def operating_threshold(m, weights=None, iou_index=0):
    """Score threshold that maximises F1 at IoU 0.5 for one class."""
    o = np.argsort(-m.score, kind="mergesort")
    w = np.ones(len(m.npig)) if weights is None else weights
    wd = w[m.det_img[o]]
    tp = np.cumsum(wd * m.tp[iou_index, o]); fp = np.cumsum(wd * m.fp[iou_index, o])
    n = float(np.dot(w, m.npig))
    f1 = 2 * tp / np.maximum(tp + fp + n, 1e-9)
    k = int(np.argmax(f1))
    return float(m.score[o][k]), float(f1[k])
