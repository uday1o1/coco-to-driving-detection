"""COCO-style evaluation with clustered bootstrap confidence intervals.

`match` reproduces pycocotools' per-image greedy matching (COCOeval.evaluateImg,
bbox, maxDets=100, crowd boxes as ignore regions) for every class, and keeps
the per-detection outcome at each IoU threshold. `ap` re-accumulates those
outcomes exactly as COCOeval.accumulate does, but with a weight per image, so
a bootstrap resample costs one pass over the detections instead of a full
re-evaluation. tests/test_evaluate.py checks that unit weights give
pycocotools' numbers to floating-point precision.

The size filter can use box height instead of area: the ground truth outside
the range becomes ignored and unmatched detections outside it are not
counted, the same rule pycocotools applies to area.
"""
from __future__ import annotations

from dataclasses import dataclass

import numba
import numpy as np

IOU_THRS = np.linspace(0.5, 0.95, 10)
REC_THRS = np.linspace(0.0, 1.0, 101)
MAX_DET = 100


@numba.njit(cache=True)
def _iou(d, g, crowd):
    """IoU of each det with each gt; crowd gts use intersection over det area."""
    out = np.zeros((d.shape[0], g.shape[0]))
    for i in range(d.shape[0]):
        da = (d[i, 2] - d[i, 0]) * (d[i, 3] - d[i, 1])
        for j in range(g.shape[0]):
            iw = min(d[i, 2], g[j, 2]) - max(d[i, 0], g[j, 0])
            ih = min(d[i, 3], g[j, 3]) - max(d[i, 1], g[j, 1])
            if iw <= 0 or ih <= 0:
                continue
            inter = iw * ih
            ga = (g[j, 2] - g[j, 0]) * (g[j, 3] - g[j, 1])
            union = da if crowd[j] else da + ga - inter
            out[i, j] = inter / union if union > 0 else 0.0
    return out


@numba.njit(cache=True)
def _match_one(d, dsize, g, gsize, crowd, lo, hi, thrs):
    """Greedy matching for one (image, class), dets already sorted by score.

    Returns det tp, det ignore (T x D), gt matched-det index at each threshold
    (T x G, -1 if none) and the gt ignore flags, in the given gt order.
    """
    G, D, T = g.shape[0], d.shape[0], thrs.shape[0]
    gig = np.zeros(G, np.bool_)
    for j in range(G):
        gig[j] = crowd[j] or gsize[j] < lo or gsize[j] > hi
    order = np.argsort(gig.astype(np.int64), kind="mergesort")
    ious = _iou(d, g[order], crowd[order])
    gig_s, crowd_s = gig[order], crowd[order]
    tp = np.zeros((T, D), np.bool_)
    dig = np.zeros((T, D), np.bool_)
    gmatch = -np.ones((T, G), np.int64)
    for t in range(T):
        taken = np.zeros(G, np.bool_)
        for i in range(D):
            best, m = min(thrs[t], 1 - 1e-10), -1
            for j in range(G):
                if taken[j] and not crowd_s[j]:
                    continue
                if m > -1 and not gig_s[m] and gig_s[j]:
                    break
                if ious[i, j] < best:
                    continue
                best, m = ious[i, j], j
            if m == -1:
                dig[t, i] = dsize[i] < lo or dsize[i] > hi
                continue
            dig[t, i] = gig_s[m]
            tp[t, i] = not gig_s[m]
            taken[m] = True
            gmatch[t, order[m]] = i
    return tp, dig, gmatch, gig


@dataclass
class Matched:
    """Per-class matching outcome, detections concatenated in image order."""
    score: np.ndarray      # D
    tp: np.ndarray         # T x D, matched to a non-ignored gt
    fp: np.ndarray         # T x D, unmatched and not ignored
    det_img: np.ndarray    # D, index into image_ids
    npig: np.ndarray       # images, non-ignored gt count
    gt_ann: np.ndarray     # G, annotation index into the gt list
    gt_img: np.ndarray     # G
    gt_ignored: np.ndarray  # G
    gt_score50: np.ndarray  # G, score of the det matched at IoU 0.5, -1 if none


def match(gt: dict, dets: dict, size="area", lo=0.0, hi=1e10, cat_ids=None):
    """Match detections to ground truth for each class.

    gt: COCO-format dict (images, annotations with bbox xywh, iscrowd, area,
    height). dets: dict of arrays image_id, boxes (xyxy), scores, category_id.
    Returns {category_id: Matched} and the ordered image id array.
    """
    image_ids = np.array(sorted(im["id"] for im in gt["images"]))
    pos = {iid: i for i, iid in enumerate(image_ids)}
    cat_ids = cat_ids or [c["id"] for c in gt["categories"]]
    anns = gt["annotations"]
    out = {}
    for c in cat_ids:
        g_by = {}
        for k, a in enumerate(anns):
            if a["category_id"] == c and a["image_id"] in pos:
                g_by.setdefault(a["image_id"], []).append(k)
        sel = np.nonzero((dets["category_id"] == c) & np.isin(dets["image_id"], image_ids))[0]
        d_img = dets["image_id"][sel]
        order = np.lexsort((np.arange(len(sel)), d_img))   # by image, insertion order kept
        sel, d_img = sel[order], d_img[order]
        starts = np.searchsorted(d_img, image_ids, "left")
        ends = np.searchsorted(d_img, image_ids, "right")
        scores, tps, fps, dimg = [], [], [], []
        npig = np.zeros(len(image_ids), np.int64)
        gt_ann, gt_img, gt_ign, gt_s50 = [], [], [], []
        for ii, iid in enumerate(image_ids):
            di = sel[starts[ii]:ends[ii]]
            di = di[np.argsort(-dets["scores"][di], kind="mergesort")][:MAX_DET]
            gk = g_by.get(iid, [])
            if not len(di) and not gk:
                continue
            gb = np.array([anns[k]["bbox"] for k in gk], np.float64).reshape(-1, 4)
            gb[:, 2:] += gb[:, :2]
            gsize = np.array([anns[k][size] for k in gk], np.float64)
            crowd = np.array([bool(anns[k]["iscrowd"]) for k in gk], np.bool_)
            db = dets["boxes"][di].astype(np.float64)
            dsize = (db[:, 2] - db[:, 0]) * (db[:, 3] - db[:, 1]) if size == "area" else db[:, 3] - db[:, 1]
            tp, dig, gm, gig = _match_one(db, dsize, gb, gsize, crowd, lo, hi, IOU_THRS)
            scores.append(dets["scores"][di]); tps.append(tp); fps.append(~tp & ~dig)
            dimg.append(np.full(len(di), ii))
            npig[ii] = int((~gig).sum())
            s = dets["scores"][di]
            gt_ann += gk; gt_img += [ii] * len(gk); gt_ign += list(gig)
            gt_s50 += [float(s[m]) if m >= 0 else -1.0 for m in gm[0]]
        T = len(IOU_THRS)
        cat = lambda xs, shape: np.concatenate(xs, axis=-1) if xs else np.zeros(shape)
        out[c] = Matched(cat(scores, (0,)).astype(np.float64), cat(tps, (T, 0)).astype(bool),
                         cat(fps, (T, 0)).astype(bool), cat(dimg, (0,)).astype(np.int64), npig,
                         np.array(gt_ann, np.int64), np.array(gt_img, np.int64),
                         np.array(gt_ign, bool), np.array(gt_s50))
    return out, image_ids


@numba.njit(cache=True)
def _ap(tp, fp, w, npig, rec_thrs):
    """COCOeval.accumulate for one class: mean precision over IoU and recall
    thresholds, plus recall at the last detection, with per-det weights."""
    T, D = tp.shape
    R = rec_thrs.shape[0]
    total = 0.0
    rec = np.zeros(T)
    for t in range(T):
        tpc = np.zeros(D)
        fpc = np.zeros(D)
        a = 0.0
        b = 0.0
        for i in range(D):
            if tp[t, i]:
                a += w[i]
            if fp[t, i]:
                b += w[i]
            tpc[i] = a
            fpc[i] = b
        rc = tpc / npig
        pr = tpc / (tpc + fpc + np.spacing(1))
        rec[t] = rc[-1] if D else 0.0
        for i in range(D - 1, 0, -1):
            if pr[i] > pr[i - 1]:
                pr[i - 1] = pr[i]
        inds = np.searchsorted(rc, rec_thrs)
        for r in range(R):
            if inds[r] < D:
                total += pr[inds[r]]
    return total / (T * R), rec


class APCache:
    """Detections of one class pre-sorted by score, ready for weighted AP."""

    def __init__(self, m: Matched):
        o = np.argsort(-m.score, kind="mergesort")
        self.tp, self.fp, self.img = m.tp[:, o], m.fp[:, o], m.det_img[o]
        self.npig = m.npig

    def ap(self, img_w=None, thr_index=None):
        """AP@[.5:.95] (or at one IoU threshold index) with per-image weights.
        Returns -1 when the class has no non-ignored ground truth."""
        w_img = np.ones(len(self.npig)) if img_w is None else img_w
        npig = float(np.dot(w_img, self.npig))
        if npig == 0:
            return -1.0
        tp, fp = (self.tp, self.fp) if thr_index is None else (self.tp[thr_index:thr_index + 1],
                                                                 self.fp[thr_index:thr_index + 1])
        return _ap(tp, fp, w_img[self.img].astype(np.float64), npig, REC_THRS)[0]


def mean_ap(caches, img_w=None, thr_index=None):
    vals = [c.ap(img_w, thr_index) for c in caches]
    vals = [v for v in vals if v >= 0]
    return float(np.mean(vals)) if vals else float("nan")


def cluster_weights(image_clusters: np.ndarray, B: int, seed: int = 0):
    """Yield per-image weights for B clustered bootstrap resamples: clusters
    are drawn with replacement and each image gets its cluster's count."""
    uniq, inv = np.unique(image_clusters, return_inverse=True)
    rng = np.random.default_rng(seed)
    for _ in range(B):
        counts = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        yield counts[inv].astype(np.float64)


def ci(samples, level=0.95):
    s = np.asarray(samples, float)
    s = s[~np.isnan(s)]
    a = (1 - level) / 2
    return float(np.quantile(s, a)), float(np.quantile(s, 1 - a))
