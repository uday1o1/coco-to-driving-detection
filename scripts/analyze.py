"""Compute every result in results/ from stored detections.

    python scripts/analyze.py main      --runs $RUNS --data $DATA
    python scripts/analyze.py audit     --runs $RUNS --data $DATA
    python scripts/analyze.py stability --runs $RUNS --data $DATA
    python scripts/analyze.py fusion    --runs $RUNS --data $DATA
    python scripts/analyze.py subgroups --runs $RUNS --data $DATA
    python scripts/analyze.py finetune  --runs $RUNS --data $DATA
    python scripts/analyze.py errors    --runs $RUNS --data $DATA
    python scripts/analyze.py gate      --runs $RUNS --data $DATA

Needs only CPU. Each section writes results/<section>.json.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sys
from multiprocessing import Pool
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
    ("dfine:ustc-community/dfine-nano-coco", "D-FINE-N", "real-time DETR"),
    ("dfine:ustc-community/dfine-medium-coco", "D-FINE-M", "real-time DETR"),
    ("dfine:ustc-community/dfine-xlarge-coco", "D-FINE-X", "real-time DETR"),
    ("gdino:IDEA-Research/grounding-dino-tiny", "Grounding DINO-T", "open vocabulary"),
    ("gdino:openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_grit_v3det", "MM-GDINO-T", "open vocabulary"),
    ("gdino:iSEE-Laboratory/llmdet_tiny", "LLMDet-T", "open vocabulary"),
]
ABLATIONS = [
    ("yolo:yolo11m.pt@1280", "YOLO11-M at 1280", "YOLO"),
    ("dfine:ustc-community/dfine-medium-coco@letterbox", "D-FINE-M letterboxed", "real-time DETR"),
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
    with Pool(procs) as p:
        return p.starmap(fn, jobs)


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
