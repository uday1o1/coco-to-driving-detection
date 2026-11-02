"""Compute every result in results/ from stored detections.

    python scripts/analyze.py main      --runs $RUNS --data $DATA
    python scripts/analyze.py audit     --runs $RUNS --data $DATA
    python scripts/analyze.py stability --runs $RUNS --data $DATA
    python scripts/analyze.py fusion    --runs $RUNS --data $DATA
    python scripts/analyze.py subgroups --runs $RUNS --data $DATA
    python scripts/analyze.py finetune  --runs $RUNS --data $DATA
    python scripts/analyze.py errors    --runs $RUNS --data $DATA
    python scripts/analyze.py gate      --runs $RUNS --data $DATA
    python scripts/analyze.py reproduction --runs $RUNS --data $DATA
    python scripts/analyze.py prompts   --runs $RUNS --data $DATA
    python scripts/analyze.py contrasts --runs $RUNS --data $DATA
    python scripts/analyze.py robustness --runs $RUNS --data $DATA

Needs only CPU. Each section writes results/<section>.json.
"""
from __future__ import annotations

import os

# One thread per worker process: the pool supplies the parallelism, and
# multi-threaded numeric libraries inside each worker oversubscribe the CPU.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import itertools
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from detbench import analysis as A  # noqa: E402
from detbench import taxonomy as tx  # noqa: E402
from detbench.evaluate import APCache, ci, cluster_weights, match, mean_ap  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MODELS = [  # spec, name, family
    ("tv:fasterrcnn_resnet50_fpn", "Faster R-CNN", "two-stage CNN"),
    ("tv:retinanet_resnet50_fpn", "RetinaNet", "one-stage CNN"),
    ("tv:fcos_resnet50_fpn", "FCOS", "one-stage CNN"),
    ("tv:fasterrcnn_resnet50_fpn_v2", "Faster R-CNN v2", "two-stage CNN"),
    ("yolo:yolo11n.pt", "YOLO11-N", "YOLO"),
    ("yolo:yolo11m.pt", "YOLO11-M", "YOLO"),
    ("yolo:yolo11x.pt", "YOLO11-X", "YOLO"),
    ("dfine:n", "D-FINE-N", "real-time DETR"),
    ("dfine:m", "D-FINE-M", "real-time DETR"),
    ("dfine:x", "D-FINE-X", "real-time DETR"),
    ("gdino:IDEA-Research/grounding-dino-tiny", "Grounding DINO-T", "open vocabulary"),
    ("gdino:openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_grit_v3det", "MM-GDINO-T", "open vocabulary"),
    ("gdino:iSEE-Laboratory/llmdet_tiny", "LLMDet-T", "open vocabulary"),
]
ABLATIONS = [
    ("yolo:yolo11m.pt@1280", "YOLO11-M at 1280", "YOLO"),
    ("dfine:m@letterbox", "D-FINE-M letterboxed", "real-time DETR"),
]
DATASETS = ["coco_val", "kitti_val", "nuimages_val"]


def tag(spec):
    return re.sub(r"[:/@]", "_", spec)


def det_path(runs, spec, ds):
    return Path(runs) / "dets" / f"{tag(spec)}__{ds}.npz"


def native_path(data, ds):
    return Path(data) / "native" / f"{ds}.json"


def save(name, obj):
    out = REPO / "results" / f"{name}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(obj, indent=1, default=float))
    print(f"wrote {out}")


def pool_map(fn, jobs, procs):
    """Parallel starmap. ProcessPoolExecutor raises if a worker is killed
    (for example by the OOM killer) instead of waiting forever."""
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(procs) as ex:
        return list(ex.map(fn, *zip(*jobs))) if jobs else []


# --------------------------------------------------------------------- main

def _main_job(runs, data, spec, name, family, ds):
    f = det_path(runs, spec, ds)
    if not f.exists():
        return None
    r = A.summarize(f, native_path(data, ds), spec, B=1000)
    return {"model": name, "spec": spec, "family": family, "dataset": ds, **r}


def section_main(a):
    jobs = [(a.runs, a.data, s, n, fam, ds) for s, n, fam in MODELS + ABLATIONS for ds in DATASETS]
    rows = [r for r in pool_map(_main_job, jobs, a.procs) if r]
    save("main", rows)


# -------------------------------------------------------------------- audit

FACTORS = ["semantics", "ignore", "minsize", "empty", "threshold", "fusion"]
TUTORIAL = dict(semantics="tutorial", ignore=False, minsize=False, empty=False, threshold=True, fusion="none")
CORRECTED = A.settings()


def _setting(bits):
    return {f: (CORRECTED[f] if b else TUTORIAL[f]) for f, b in zip(FACTORS, bits)}


def _audit_job(runs, data, spec, ds, bits):
    f = det_path(runs, spec, ds)
    if not f.exists():
        return None
    _, gt, dets = A.prepare(f, native_path(data, ds), spec, _setting(bits))
    return {"spec": spec, "dataset": ds, "bits": bits, "AP": A.coco_stats(gt, dets)["AP"]}


def shapley(values: dict, n: int):
    """Exact Shapley attribution of v(all corrected) - v(all tutorial)."""
    out = []
    for i in range(n):
        tot = 0.0
        for bits in itertools.product((0, 1), repeat=n):
            if bits[i]:
                continue
            k = sum(bits)
            with_i = tuple(1 if j == i else b for j, b in enumerate(bits))
            tot += math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n) * (values[with_i] - values[bits])
        out.append(tot)
    return out


def section_audit(a):
    combos = list(itertools.product((0, 1), repeat=len(FACTORS)))
    jobs = [(a.runs, a.data, s, ds, bits) for s, _, _ in MODELS for ds in DATASETS for bits in combos]
    rows = [r for r in pool_map(_audit_job, jobs, a.procs) if r]
    summary = []
    for s, name, fam in MODELS:
        for ds in DATASETS:
            v = {tuple(r["bits"]): r["AP"] for r in rows if r["spec"] == s and r["dataset"] == ds}
            if len(v) < len(combos):
                continue
            ones, zeros = tuple([1] * len(FACTORS)), tuple([0] * len(FACTORS))
            main_effects = {}
            for i, f in enumerate(FACTORS):
                on = [x for b, x in v.items() if b[i]]
                off = [x for b, x in v.items() if not b[i]]
                main_effects[f] = float(np.mean(on) - np.mean(off))
            summary.append({"model": name, "spec": s, "family": fam, "dataset": ds,
                            "tutorial_AP": v[zeros], "corrected_AP": v[ones],
                            "shapley": dict(zip(FACTORS, shapley(v, len(FACTORS)))),
                            "main_effects": main_effects})
    save("audit", {"factors": FACTORS, "tutorial": TUTORIAL, "corrected": CORRECTED,
                   "cells": rows, "summary": summary})


# ---------------------------------------------------------------- stability

def kendall_tau(a, b):
    from scipy.stats import kendalltau
    return float(kendalltau(a, b).statistic)


def _caches(runs, data, spec, ds, s):
    _, gt, dets = A.prepare(det_path(runs, spec, ds), native_path(data, ds), spec, s)
    m, ids = match(gt, dets)
    return [APCache(m[c]) for c in m], ids


def section_stability(a):
    """How stable is the model ranking when only 120 images are evaluated?"""
    out = {}
    for ds in ("kitti_val", "nuimages_val"):
        for proto, s in (("tutorial", TUTORIAL), ("corrected", CORRECTED)):
            res = pool_map(_caches, [(a.runs, a.data, sp, ds, s) for sp, _, _ in MODELS], a.procs)
            ids = res[0][1]
            full = [mean_ap(c) for c, _ in res]
            rng = np.random.default_rng(0)
            taus, spreads = [], []
            for _ in range(500):
                w = np.zeros(len(ids)); w[rng.choice(len(ids), 120, replace=False)] = 1
                sub = [mean_ap(c, w) for c, _ in res]
                taus.append(kendall_tau(full, sub)); spreads.append(sub)
            spreads = np.array(spreads)
            out[f"{ds}/{proto}"] = {
                "models": [n for _, n, _ in MODELS], "full_AP": full,
                "n120_ci95": [ci(spreads[:, i]) for i in range(len(MODELS))],
                "kendall_tau_median": float(np.median(taus)), "kendall_tau_ci95": ci(taus)}
    save("stability", out)


# ------------------------------------------------------------------- fusion

TUNE_SPEC = "tv:fasterrcnn_resnet50_fpn_v2"


def _tune_job(runs, data, fusion, ioa, bike_min, stretch):
    _, gt, dets = A.prepare(det_path(runs, TUNE_SPEC, "kitti_train"), native_path(data, "kitti_train"),
                            TUNE_SPEC, A.settings(fusion=fusion), ioa, bike_min, stretch)
    m, _ = match(gt, dets, cat_ids=[2, 3])
    ap = {tx.CLASSES[c - 1]: APCache(m[c]).ap() for c in m}
    return {"fusion": fusion, "ioa": ioa, "bike_min": bike_min, "stretch": stretch,
            **ap, "objective": (ap["person"] + ap["cyclist"]) / 2}


def _rider_decisions(runs, data, spec, min_pair_score=0.3):
    """On nuImages, how often does fusion confidently pair a detected bicycle
    with a person (pair score >= min_pair_score), split by the ground-truth
    rider attribute? Pairs below that score only add low-ranked cyclist boxes."""
    from detbench import protocol
    from detbench.evaluate import _iou
    nat = A.native(str(native_path(data, "nuimages_val")))
    d = protocol.load(det_path(runs, spec, "nuimages_val"))
    names = np.array([str(n).lower() for n in d["names"]])
    labels = names[d["labels"]]
    bikes = {}
    for b in nat["boxes"]:
        if b["fine"].startswith("vehicle.bicycle."):
            bikes.setdefault(b["image_id"], []).append(b)
    counts = {"with_rider": [0, 0], "without_rider": [0, 0]}   # [detected, paired]
    order = np.argsort(d["image_id"], kind="mergesort")
    img = d["image_id"][order]
    for iid, gts in bikes.items():
        sl = order[np.searchsorted(img, iid, "left"):np.searchsorted(img, iid, "right")]
        if not len(sl):
            continue
        sc = d["scores"][sl]
        _, _, _, pairs = protocol.fuse_riders(d["boxes"][sl], sc, labels[sl],
                                              protocol.FUSION["ioa"], protocol.FUSION["bike_min"], True)
        paired = {j for i, j in pairs if np.sqrt(sc[i] * sc[j]) >= min_pair_score}
        bk = [k for k in range(len(sl)) if labels[sl][k] == "bicycle" and d["scores"][sl][k] >= protocol.FUSION["bike_min"]]
        if not bk:
            continue
        for g in gts:
            kind = g["fine"].split(".")[-1]
            gb = np.array([g["xyxy"]], np.float64)
            # with a rider the GT box covers rider and bike; the bike part is its lower half
            if kind == "with_rider":
                gb[0, 1] = (gb[0, 1] + gb[0, 3]) / 2
            ious = _iou(d["boxes"][sl][bk].astype(np.float64), gb, np.zeros(1, np.bool_))[:, 0]
            if ious.max() >= 0.5:
                counts[kind][0] += 1
                counts[kind][1] += int(bk[int(ious.argmax())] in paired)
    return {k: {"detected": v[0], "paired": v[1], "rate": v[1] / max(v[0], 1)} for k, v in counts.items()}


def section_fusion(a):
    grid = ([("rider", i, b, 1.0) for i in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
             for b in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
            + [("stretch", 0.3, 0.5, f) for f in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)]
            + [("none", 0.3, 0.5, 1.0)])
    tune = pool_map(_tune_job, [(a.runs, a.data, *g) for g in grid], a.procs)
    best = {f: max((t for t in tune if t["fusion"] == f), key=lambda t: t["objective"]) for f in ("rider", "stretch")}
    from detbench import protocol
    chosen = {"ioa": best["rider"]["ioa"], "bike_min": best["rider"]["bike_min"]}
    if chosen != protocol.FUSION:
        print(f"set protocol.FUSION = {chosen} in detbench/protocol.py and re-run")
    held_out = []
    jobs = []
    for s, name, fam in MODELS:
        for ds in ("kitti_val", "nuimages_val"):
            for fusion in ("none", "stretch", "rider"):
                jobs.append((a.runs, a.data, s, name, ds, fusion, best["stretch"]["stretch"]))
    held_out = [r for r in pool_map(_fusion_eval_job, jobs, a.procs) if r]
    decisions = {name: _rider_decisions(a.runs, a.data, s) for s, name, fam in MODELS
                 if det_path(a.runs, s, "nuimages_val").exists() and not s.startswith("gdino")}
    save("fusion", {"tuned_on": "kitti_train / " + TUNE_SPEC, "grid": tune, "best": best,
                    "held_out": held_out, "rider_decisions_nuimages": decisions})


def _fusion_eval_job(runs, data, spec, name, ds, fusion, stretch):
    f = det_path(runs, spec, ds)
    if not f.exists():
        return None
    _, gt, dets = A.prepare(f, native_path(data, ds), spec, A.settings(fusion=fusion), stretch=stretch)
    m, _ = match(gt, dets)
    c = {tx.CLASSES[k - 1]: APCache(m[k]) for k in m}
    return {"model": name, "dataset": ds, "fusion": fusion, "AP": mean_ap(c.values()),
            **{k: v.ap() for k, v in c.items()}}


# ---------------------------------------------------------------- subgroups

PERSON_GROUPS = ["human.pedestrian.adult", "human.pedestrian.child",
                 "human.pedestrian.construction_worker", "human.pedestrian.police_officer"]
SG_HEIGHT_BINS = [(0, 40), (40, 80), (80, 1e10)]
FT_MODELS = [(f"ft_yolo11m_seed{s}", f"YOLO11-M fine-tuned (seed {s})", "YOLO") for s in range(3)] + \
            [(f"ft_frcnn_v2_seed{s}", f"Faster R-CNN v2 fine-tuned (seed {s})", "two-stage CNN") for s in range(3)]


def _group_recall(found, groups, heights, w_obj, std_ref=None):
    """Weighted recall per group, and the child/adult recall standardized to
    the adult height distribution."""
    out = {}
    for g in PERSON_GROUPS:
        s = groups == g
        out[g] = float(np.dot(w_obj[s], found[s]) / max(w_obj[s].sum(), 1e-9))
    ref = groups == "human.pedestrian.adult"
    bins = np.digitize(heights, [b[1] for b in SG_HEIGHT_BINS[:-1]])
    share = np.array([w_obj[ref & (bins == k)].sum() for k in range(len(SG_HEIGHT_BINS))])
    share = share / share.sum()
    for g in PERSON_GROUPS[1:]:
        r = []
        for k in range(len(SG_HEIGHT_BINS)):
            s = (groups == g) & (bins == k)
            r.append(np.dot(w_obj[s], found[s]) / w_obj[s].sum() if w_obj[s].sum() > 0 else np.nan)
        r = np.array(r)
        ok = ~np.isnan(r)
        out[g + "/height_std"] = float(np.dot(share[ok], r[ok]) / share[ok].sum())
    return out


SG_B = 20000   # enough resamples for Holm-corrected p-values over the test family


def _threshold_fn(m):
    """Fast F1-optimal threshold for weighted resamples (order computed once)."""
    o = np.argsort(-m.score, kind="mergesort")
    sc, tp, fp, img = m.score[o], m.tp[0, o].astype(float), m.fp[0, o].astype(float), m.det_img[o]

    def f(w):
        wd = w[img]
        t, p = np.cumsum(wd * tp), np.cumsum(wd * fp)
        f1 = 2 * t / np.maximum(t + p + float(np.dot(w, m.npig)), 1e-9)
        return float(sc[int(np.argmax(f1))])
    return f


def _boot_p(gap):
    """Two-sided bootstrap p-value with the (k + 1) / (B + 1) correction."""
    g = np.asarray(gap)
    k = min((g <= 0).sum(), (g >= 0).sum())
    return float(min(1.0, 2 * (k + 1) / (len(g) + 1)))


def _subgroup_job(runs, data, spec, name):
    f = det_path(runs, spec, "nuimages_val")
    if not f.exists():
        return None
    nat, gt, dets = A.prepare(f, native_path(data, "nuimages_val"), spec, A.settings())
    m, ids = match(gt, dets, cat_ids=[2])
    p = m[2]
    anns, boxes = gt["annotations"], nat["boxes"]
    keep = ~p.gt_ignored
    groups = np.array([boxes[anns[k]["native_index"]]["fine"] for k in p.gt_ann])[keep]
    heights = np.array([anns[k]["height"] for k in p.gt_ann])[keep]
    score50, obj_img = p.gt_score50[keep], p.gt_img[keep]
    cl = {im["id"]: im["cluster"] for im in nat["images"]}
    clusters = np.array([cl[i] for i in ids])
    res = {"model": name, "n": {g: int((groups == g).sum()) for g in PERSON_GROUPS}}
    thr_fn = _threshold_fn(p)
    thr = thr_fn(np.ones(len(ids)))
    res["threshold"] = thr
    res["recall"] = _group_recall(score50 >= thr, groups, heights, np.ones(len(groups)))
    boot = []
    for w in cluster_weights(clusters, SG_B, seed=2):
        t = thr_fn(w)                                   # threshold re-estimated
        boot.append(_group_recall(score50 >= t, groups, heights, w[obj_img]))
    res["ci95"] = {k: ci([b[k] for b in boot]) for k in boot[0]}
    for g in PERSON_GROUPS[1:]:
        for suffix in ("", "/height_std"):
            gap = [b["human.pedestrian.adult"] - b[g + suffix] for b in boot]
            res.setdefault("gap_vs_adult", {})[g + suffix] = {
                "gap": res["recall"]["human.pedestrian.adult"] - res["recall"][g + suffix],
                "ci95": ci(gap), "p_two_sided": _boot_p(gap)}
    res["heights_median"] = {g: float(np.median(heights[groups == g])) for g in PERSON_GROUPS}
    res["bootstrap_resamples"] = SG_B
    return res


def holm(pvals):
    order = np.argsort(pvals)
    adj, running = np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj.tolist()


def section_subgroups(a):
    specs = [(s, n) for s, n, _ in MODELS] + [(s, n) for s, n, _ in FT_MODELS]
    rows = [r for r in pool_map(_subgroup_job, [(a.runs, a.data, s, n) for s, n in specs], a.procs) if r]
    # Holm within two pre-declared families: the zero-shot models (child and
    # construction worker, raw and at matched height) and the fine-tuned models.
    tested = ("human.pedestrian.child", "human.pedestrian.child/height_std",
              "human.pedestrian.construction_worker", "human.pedestrian.construction_worker/height_std")
    for family in (lambda r: "fine-tuned" not in r["model"], lambda r: "fine-tuned" in r["model"]):
        keys = [(i, k) for i, r in enumerate(rows) if family(r) for k in tested]
        for (i, k), p in zip(keys, holm([rows[i]["gap_vs_adult"][k]["p_two_sided"] for i, k in keys])):
            rows[i]["gap_vs_adult"][k]["p_holm"] = p
    save("subgroups", {"operating_point": "F1-optimal score threshold for person at IoU 0.5, "
                                          "re-estimated in every bootstrap resample",
                       "bootstrap_resamples": SG_B,
                       "holm_families": ["zero-shot models: child and construction worker, raw and at matched height",
                                         "fine-tuned models: same four tests"],
                       "height_bins": SG_HEIGHT_BINS, "rows": rows})


# ----------------------------------------------------------------- finetune

def section_finetune(a):
    """Fine-tuned (3 seeds) against the same architecture zero-shot."""
    pairs = [("YOLO11-M", "yolo:yolo11m.pt", "ft_yolo11m_seed"),
             ("Faster R-CNN v2", "tv:fasterrcnn_resnet50_fpn_v2", "ft_frcnn_v2_seed")]
    jobs = [(a.runs, a.data, f"{pre}{s}", f"{name} seed {s}", name, ds)
            for name, _, pre in pairs for s in range(3) for ds in DATASETS]
    rows = [r for r in pool_map(_main_job, jobs, a.procs) if r]
    diffs = []
    for name, zs, pre in pairs:
        for ds in DATASETS:
            for s in range(3):
                fa, fb = det_path(a.runs, f"{pre}{s}", ds), det_path(a.runs, zs, ds)
                if fa.exists() and fb.exists():
                    diffs.append({"model": name, "dataset": ds, "seed": s,
                                  **A.paired_diff(fa, fb, native_path(a.data, ds), f"{pre}{s}", zs)})
    save("finetune", {"runs": rows, "diff_vs_zero_shot": diffs})


# ------------------------------------------------------------------- errors

def _error_job(runs, data, spec, name, ds):
    """At each class's F1-optimal threshold (IoU 0.5), sort the evaluator's own
    false positives (ignore regions and size rules already applied) into
    duplicate / class confusion / localisation / background, following the
    TIDE categories, and count misses by object height."""
    from detbench.evaluate import _iou
    f = det_path(runs, spec, ds)
    if not f.exists():
        return None
    nat, gt, dets = A.prepare(f, native_path(data, ds), spec, A.settings())
    m, ids = match(gt, dets)
    anns = gt["annotations"]
    real = {}
    for k, an in enumerate(anns):
        if not an["iscrowd"]:
            x, y, w, h = an["bbox"]
            real.setdefault(an["image_id"], []).append((an["category_id"], [x, y, x + w, y + h]))
    out = {"model": name, "dataset": ds, "classes": {}}
    for c, mc in m.items():
        if mc.npig.sum() == 0:
            continue
        thr, f1 = A.operating_threshold(mc)
        kept = mc.score >= thr
        fp = np.nonzero(kept & mc.fp[0])[0]
        types = {"duplicate": 0, "confusion": 0, "localization": 0, "background": 0}
        for i in fp:
            box = dets["boxes"][mc.det_idx[i]][None].astype(np.float64)
            gts = real.get(int(dets["image_id"][mc.det_idx[i]]), [])
            same = np.array([g for cc, g in gts if cc == c], np.float64).reshape(-1, 4)
            other = np.array([g for cc, g in gts if cc != c], np.float64).reshape(-1, 4)
            iou_s = _iou(box, same, np.zeros(len(same), np.bool_))[0] if len(same) else np.zeros(0)
            iou_o = _iou(box, other, np.zeros(len(other), np.bool_))[0] if len(other) else np.zeros(0)
            if len(iou_s) and iou_s.max() >= 0.5:
                types["duplicate"] += 1          # overlaps a box that a higher-scoring detection took
            elif len(iou_o) and iou_o.max() >= 0.5:
                types["confusion"] += 1
            elif len(iou_s) and iou_s.max() >= 0.1:
                types["localization"] += 1
            else:
                types["background"] += 1
        h = np.array([anns[k]["height"] for k in mc.gt_ann])
        keep = ~mc.gt_ignored
        missed = keep & (mc.gt_score50 < thr)
        bins = {f"{lo}-{int(hi) if hi < 1e9 else 'inf'}": [int((missed & (h >= lo) & (h < hi)).sum()),
                                                            int((keep & (h >= lo) & (h < hi)).sum())]
                for lo, hi in A.HEIGHT_BINS}
        out["classes"][tx.CLASSES[c - 1]] = {"threshold": thr, "f1": f1, "tp": int((kept & mc.tp[0]).sum()),
                                             "fp": int(len(fp)), "fp_types": types, "missed_by_height": bins}
    return out


def section_errors(a):
    jobs = [(a.runs, a.data, s, n, ds) for s, n, _ in MODELS for ds in DATASETS]
    save("errors", [r for r in pool_map(_error_job, jobs, a.procs) if r])


# --------------------------------------------------------------------- gate

PUBLISHED_COCO_AP = {  # COCO val2017 box AP from the model's own documentation or paper
    "tv:fasterrcnn_resnet50_fpn": 37.0, "tv:retinanet_resnet50_fpn": 36.4,
    "tv:fcos_resnet50_fpn": 39.2, "tv:fasterrcnn_resnet50_fpn_v2": 46.7,
    "yolo:yolo11n.pt": 39.5, "yolo:yolo11m.pt": 51.5, "yolo:yolo11x.pt": 54.7,
    "dfine:n": 42.8, "dfine:m": 52.3, "dfine:x": 55.8,
    "hfdfine:ustc-community/dfine-nano-coco": 42.8, "hfdfine:ustc-community/dfine-medium-coco": 52.3,
    "hfdfine:ustc-community/dfine-xlarge-coco": 55.8,
    "gdino:IDEA-Research/grounding-dino-tiny@coco80": 48.4,
    "yolo:yolo11n.pt@single": 39.5,
    "gdino:openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_grit_v3det@coco80": 50.4,
    "gdino:iSEE-Laboratory/llmdet_tiny@coco80": 55.6,   # LLMDet paper, Table 4; trained with COCO images
}


def section_gate(a):
    from detbench.coco_gate import coco_ap
    ann = Path(a.data) / "coco/annotations/instances_val2017.json"
    rows = []
    hf = [(f"hfdfine:ustc-community/dfine-{v}-coco", f"D-FINE-{k} (Hugging Face port)", "real-time DETR")
          for k, v in (("N", "nano"), ("M", "medium"), ("X", "xlarge"))]
    extra = [("yolo:yolo11n.pt@single", "YOLO11-N, single-label NMS (prediction mode)", "YOLO")]
    for s, name, fam in MODELS + hf + extra:
        spec = s + "@coco80" if s.startswith("gdino") else s
        f = det_path(a.runs, spec, "coco_val")
        if f.exists():
            r = coco_ap(f, ann)
            rows.append({"model": name, "spec": spec, "measured_AP": round(100 * r["AP"], 1),
                         "published_AP": PUBLISHED_COCO_AP.get(spec), **r})
            if spec.startswith("hfdfine"):        # the same file mapped by name without the VOC aliases
                rows[-1]["AP_without_voc_aliases"] = coco_ap(f, ann, aliases=False)["AP"]
            print(rows[-1]["model"], rows[-1]["measured_AP"], rows[-1]["published_AP"])
    save("gate", rows)


# --------------------------------------------------------------- robustness

ROBUST_PAIRS = [("tv:fcos_resnet50_fpn", "tv:fasterrcnn_resnet50_fpn"),
                ("tv:retinanet_resnet50_fpn", "tv:fasterrcnn_resnet50_fpn"),
                ("yolo:yolo11x.pt", "tv:fasterrcnn_resnet50_fpn_v2"),
                ("yolo:yolo11x.pt", "dfine:x"),
                ("gdino:IDEA-Research/grounding-dino-tiny", "tv:fasterrcnn_resnet50_fpn_v2")]


def _class_contrast_job(runs, data, a_spec, b_spec, ds, cls, fusion):
    s = A.settings(fusion=fusion)
    r = A.paired_diff(det_path(runs, a_spec, ds), det_path(runs, b_spec, ds), native_path(data, ds),
                      a_spec, b_spec, s, s, cls=cls)
    return {"a": a_spec, "b": b_spec, "dataset": ds, "class": tx.CLASSES[cls - 1], "fusion": fusion, **r}


def _unified_nms(dets, thr=0.7):
    import torch, torchvision
    keep = []
    order = np.argsort(dets["image_id"], kind="mergesort")
    img = dets["image_id"][order]
    for sl in np.split(order, np.flatnonzero(np.diff(img)) + 1):
        if len(sl):
            k = torchvision.ops.batched_nms(torch.tensor(dets["boxes"][sl], dtype=torch.float32),
                                            torch.tensor(dets["scores"][sl], dtype=torch.float32),
                                            torch.tensor(dets["category_id"][sl]), thr).numpy()
            keep.append(sl[k])
    keep = np.sort(np.concatenate(keep))
    return {k: v[keep] for k, v in dets.items()}


def _nms_job(runs, data, spec, ds):
    _, gt, dets = A.prepare(det_path(runs, spec, ds), native_path(data, ds), spec, A.settings())
    out = {"spec": spec, "dataset": ds}
    for key, d in (("", dets), ("_unified_nms_0.7", _unified_nms(dets))):
        m, _ = match(gt, d)
        out["AP" + key] = A.coco_stats(gt, d)["AP"]
        out["per_class" + key] = {tx.CLASSES[k - 1]: APCache(m[k]).ap() for k in m}
    return out


NMS_SPECS = ("yolo:yolo11x.pt", "dfine:x", "tv:fasterrcnn_resnet50_fpn_v2", "tv:retinanet_resnet50_fpn",
             "gdino:IDEA-Research/grounding-dino-tiny",
             "gdino:openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_grit_v3det",
             "gdino:iSEE-Laboratory/llmdet_tiny")


def nms_check(a):
    """Class-wise NMS (IoU 0.7) after mapping to the unified classes: removes
    boxes that two source classes or two prompt terms both produced."""
    return pool_map(_nms_job, [(a.runs, a.data, s, ds) for s in NMS_SPECS for ds in ("kitti_val", "nuimages_val")],
                    a.procs)


def section_unified_nms(a):
    """Re-run only the NMS check and update robustness.json in place."""
    f = REPO / "results" / "robustness.json"
    out = json.loads(f.read_text())
    out["unified_nms"] = nms_check(a)
    save("robustness", out)


def section_robustness(a):
    """Ranking agreement on matched classes, per-class contrasts with and without
    rider fusion, NMS after class merging, and YOLO's NMS mode under the
    tutorial protocol."""
    from scipy.stats import kendalltau, spearmanr
    out = {}
    # 1. ranking agreement between datasets on matched classes, with a bootstrap over models
    rows = json.loads((REPO / "results" / "main.json").read_text())
    get = {(r["spec"], r["dataset"]): r for r in rows}
    specs = [s for s, _, _ in MODELS]
    agree = {}
    for basis, classes in (("vehicle", ["vehicle"]), ("person", ["person"]), ("vehicle+person", ["vehicle", "person"])):
        v = {ds: np.array([np.mean([get[(s, ds)]["per_class"][c]["AP"] for c in classes]) for s in specs])
             for ds in DATASETS}
        rng = np.random.default_rng(0)
        res = {}
        for x, y in (("coco_val", "kitti_val"), ("coco_val", "nuimages_val"), ("kitti_val", "nuimages_val")):
            boot = []
            for _ in range(5000):
                i = rng.integers(0, len(specs), len(specs))
                if len(set(i)) > 2:
                    boot.append(spearmanr(v[x][i], v[y][i]).statistic)
            res[f"{x}~{y}"] = {"spearman": float(spearmanr(v[x], v[y]).statistic),
                               "kendall": float(kendalltau(v[x], v[y]).statistic), "spearman_ci95": ci(boot)}
        for ds in ("kitti_val", "nuimages_val"):
            slope = np.polyfit(v["coco_val"], v[ds], 1)[0]
            bs = []
            for _ in range(5000):
                i = rng.integers(0, len(specs), len(specs))
                if np.ptp(v["coco_val"][i]) > 0:
                    bs.append(np.polyfit(v["coco_val"][i], v[ds][i], 1)[0])
            res[f"slope_{ds}_on_coco"] = {"slope": float(slope), "ci95": ci(bs)}
        agree[basis] = res
    out["ranking_agreement"] = agree
    # 2. architecture contrasts per class, with and without rider fusion
    jobs = [(a.runs, a.data, x, y, ds, cls, fu) for x, y in ROBUST_PAIRS for ds in ("kitti_val", "nuimages_val")
            for cls in (1, 2) for fu in ("rider", "none")]
    jobs += [(a.runs, a.data, x, y, "coco_val", cls, "none") for x, y in ROBUST_PAIRS for cls in (1, 2)]
    out["class_contrasts"] = pool_map(_class_contrast_job, jobs, a.procs)
    # 3. NMS after merging COCO classes into the unified ones
    out["unified_nms"] = nms_check(a)
    # 4. YOLO NMS mode under the tutorial protocol, same 500 nuImages frames
    nm = {}
    for spec, ds in (("yolo:yolo11x.pt@single", "nuimages_ngdet500"), ("yolo:yolo11x.pt", "nuimages_val")):
        f = det_path(a.runs, spec, ds)
        if f.exists():
            _, gt, dets = A.prepare(f, native_path(a.data, "nuimages_ngdet500"), spec, TUTORIAL)
            nm[spec] = A.coco_stats(gt, dets)["AP"]
    out["yolo_nms_mode_tutorial_protocol_nuimages500"] = nm
    save("robustness", out)


# ---------------------------------------------------------------- contrasts

CONTRASTS = [  # (a, b, datasets): clustered paired bootstrap of AP(a) - AP(b)
    ("yolo:yolo11m.pt@1280", "yolo:yolo11m.pt", ("kitti_val", "nuimages_val")),
    ("dfine:m@letterbox", "dfine:m", ("kitti_val", "nuimages_val")),
    ("gdino:iSEE-Laboratory/llmdet_tiny",
     "gdino:openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_grit_v3det", tuple(DATASETS)),
    ("gdino:IDEA-Research/grounding-dino-tiny", "tv:fasterrcnn_resnet50_fpn_v2", ("kitti_val", "nuimages_val")),
    ("yolo:yolo11x.pt", "dfine:x", tuple(DATASETS)),
    ("yolo:yolo11x.pt", "tv:fasterrcnn_resnet50_fpn_v2", tuple(DATASETS)),
    ("tv:fcos_resnet50_fpn", "tv:fasterrcnn_resnet50_fpn", tuple(DATASETS)),
    ("tv:retinanet_resnet50_fpn", "tv:fasterrcnn_resnet50_fpn", tuple(DATASETS)),
    ("tv:fasterrcnn_resnet50_fpn_v2", "tv:fasterrcnn_resnet50_fpn", tuple(DATASETS)),
]


def _contrast_job(runs, data, a, b, ds):
    fa, fb = det_path(runs, a, ds), det_path(runs, b, ds)
    if not (fa.exists() and fb.exists()):
        return None
    return {"a": a, "b": b, "dataset": ds, **A.paired_diff(fa, fb, native_path(data, ds), a, b)}


def section_contrasts(a):
    jobs = [(a.runs, a.data, x, y, ds) for x, y, dss in CONTRASTS for ds in dss]
    save("contrasts", [r for r in pool_map(_contrast_job, jobs, a.procs) if r])


# ------------------------------------------------------------------ prompts

def _prompt_job(runs, data, spec, name, prompt, ds):
    """Open-vocabulary AP for one prompt set. The default prompt set
    (synonyms) comes from the main run, restricted to the same images."""
    f = det_path(runs, spec if prompt == "synonyms" else f"{spec}@{prompt}",
                 "nuimages_val" if (prompt == "synonyms" and ds == "nuimages_val_sub") else ds)
    if not f.exists():
        return None
    _, gt, dets = A.prepare(f, native_path(data, ds), spec, A.settings())
    m, _ = match(gt, dets)
    c = {tx.CLASSES[k - 1]: APCache(m[k]) for k in m}
    return {"model": name, "prompt": prompt, "dataset": ds, "AP": mean_ap(c.values()),
            **{k: v.ap() for k, v in c.items()}}


def section_prompts(a):
    jobs = [(a.runs, a.data, s, n, p, ds) for s, n, fam in MODELS if fam == "open vocabulary"
            for p in ("synonyms", "bare", "rider") for ds in ("kitti_val", "nuimages_val_sub")]
    save("prompts", [r for r in pool_map(_prompt_job, jobs, a.procs) if r])


# ------------------------------------------------------------- reproduction

NG_TABLE = {  # ngdet TUTORIAL.md section 12.1, mAP@[.5:.95]
    ("torchvision:fasterrcnn_resnet50_fpn_v2", "coco"): 0.581,
    ("torchvision:fasterrcnn_resnet50_fpn_v2", "kitti_train120"): 0.279,
    ("torchvision:fasterrcnn_resnet50_fpn_v2", "nuimages"): 0.299,
    ("yolo:yolo11x.pt", "coco"): 0.573, ("yolo:yolo11x.pt", "kitti_train120"): 0.260,
    ("yolo:yolo11x.pt", "nuimages"): 0.249,
    ("gdino:IDEA-Research/grounding-dino-tiny", "coco"): 0.527,
    ("gdino:IDEA-Research/grounding-dino-tiny", "kitti_train120"): 0.223,
    ("gdino:IDEA-Research/grounding-dino-tiny", "nuimages"): 0.306,
}


def section_reproduction(a):
    """ngdet's own numbers (scripts/reproduce_tutorial.sh) against the
    tutorial table, and this repo's tutorial-mode protocol on the same
    images and the same per-image inference."""
    rows = []
    ng = Path(a.runs) / "ngdet_out"
    if not ng.exists():
        ng = REPO / "logs" / "ngdet"          # committed copy of the ngdet outputs
    for d in ("k_c", "nuim", "k_train120"):
        for m in json.loads((ng / d / "metrics.json").read_text()):
            ds = m["_meta"]["dataset"] + ("_train120" if d == "k_train120" else "")
            rows.append({"model": m["_meta"]["model"], "dataset": ds, "images": m["_meta"]["num_images"],
                         "ngdet_AP": m["mAP"], "tutorial_table_AP": NG_TABLE.get((m["_meta"]["model"], ds)),
                         "per_class": m["per_class"]})
    ours = {"kitti": "kitti_ngdet120", "kitti_train120": "kitti_train120",
            "coco": "coco_first120", "nuimages": "nuimages_ngdet500"}
    specs = {"torchvision:fasterrcnn_resnet50_fpn_v2": "tv:fasterrcnn_resnet50_fpn_v2",
             "yolo:yolo11x.pt": "yolo:yolo11x.pt@single"}
    for r in rows:
        if r["model"] in specs:
            ds = ours[r["dataset"]]
            _, gt, dets = A.prepare(det_path(a.runs, specs[r["model"]], ds), native_path(a.data, ds),
                                    specs[r["model"]], TUTORIAL)
            r["this_repo_tutorial_mode_AP"] = A.coco_stats(gt, dets)["AP"]
    save("reproduction", {"ngdet_commit": "e347f6ff4468cc33442f38277119bb7bdc18a8d1",
                          "note": "kitti = ngdet's own seed-42 split; kitti_train120 = first 120 images of "
                                  "ImageSets/train.txt, the images behind the tutorial table", "rows": rows})


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("section")
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--procs", type=int, default=16)
    a = p.parse_args()
    globals()[f"section_{a.section}"](a)


if __name__ == "__main__":
    main()
