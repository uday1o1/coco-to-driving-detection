"""The fast evaluator must give pycocotools' numbers exactly.

Synthetic scenes with crowd (ignore) boxes, duplicates, score ties, empty
images and every COCO area range. Run: python -m pytest tests -q
"""
import contextlib
import io

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from detbench.evaluate import APCache, match, mean_ap

AREAS = {"all": (0, 1e10), "small": (0, 32 ** 2), "medium": (32 ** 2, 96 ** 2), "large": (96 ** 2, 1e10)}


def scene(seed, n_img=60, decimals=2):
    rng = np.random.default_rng(seed)
    images = [{"id": i + 1, "width": 640, "height": 480} for i in range(n_img)]
    anns, dets = [], []
    for im in images:
        for _ in range(rng.integers(0, 8)):
            c = int(rng.integers(1, 4))
            w, h = rng.uniform(4, 200, 2)
            x, y = rng.uniform(0, 400), rng.uniform(0, 300)
            crowd = int(rng.random() < 0.1)
            anns.append({"id": len(anns) + 1, "image_id": im["id"], "category_id": c,
                         "bbox": [x, y, w, h], "area": w * h, "iscrowd": crowd, "height": h})
            for _ in range(rng.integers(0, 3)):           # noisy hits and duplicates
                j = rng.normal(0, 0.15 * min(w, h), 4)
                s = round(float(rng.random()), decimals)   # coarse scores create ties
                dets.append([im["id"], x + j[0], y + j[1], max(1, w + j[2]), max(1, h + j[3]), s,
                             c if rng.random() > 0.1 else int(rng.integers(1, 4))])
        for _ in range(rng.integers(0, 5)):               # background false positives
            dets.append([im["id"], *rng.uniform(0, 400, 2), *rng.uniform(5, 150, 2),
                         round(float(rng.random()) * 0.6, decimals), int(rng.integers(1, 4))])
    gt = {"images": images, "annotations": anns,
          "categories": [{"id": c, "name": str(c)} for c in (1, 2, 3)]}
    return gt, np.array(dets)


def reference(gt, rows, area):
    with contextlib.redirect_stdout(io.StringIO()):
        g = COCO()
        g.dataset = gt
        g.createIndex()
        E = COCOeval(g, g.loadRes(rows), "bbox")
        E.evaluate(); E.accumulate()
    a = list(E.params.areaRngLbl).index(area)
    p = E.eval["precision"][:, :, :, a, -1]
    per_class = [p[:, :, k][p[:, :, k] > -1].mean() if (p[:, :, k] > -1).any() else -1
                 for k in range(p.shape[2])]
    return np.mean(p[p > -1]), per_class


def ours(gt, rows, area):
    dets = {"image_id": rows[:, 0].astype(int),
            "boxes": np.column_stack([rows[:, 1], rows[:, 2], rows[:, 1] + rows[:, 3], rows[:, 2] + rows[:, 4]]),
            "scores": rows[:, 5], "category_id": rows[:, 6].astype(int)}
    m, _ = match(gt, dets, "area", *AREAS[area])
    caches = [APCache(m[c]) for c in (1, 2, 3)]
    return mean_ap(caches), [c.ap() for c in caches]


def test_matches_pycocotools():
    for seed in range(5):
        gt, rows = scene(seed)
        for area in AREAS:
            ref, ref_k = reference(gt, rows, area)
            got, got_k = ours(gt, rows, area)
            assert abs(ref - got) < 1e-12, (seed, area, ref, got)
            assert np.allclose(ref_k, got_k, atol=1e-12), (seed, area)


def test_bootstrap_weights_equal_duplication():
    """Weight 2 on an image equals evaluating with that image copied. Scores
    are continuous here: with ties, the copy's position among equal scores
    differs from the weighted version's."""
    gt, rows = scene(7, n_img=20, decimals=12)
    dets = {"image_id": rows[:, 0].astype(int),
            "boxes": np.column_stack([rows[:, 1], rows[:, 2], rows[:, 1] + rows[:, 3], rows[:, 2] + rows[:, 4]]),
            "scores": rows[:, 5], "category_id": rows[:, 6].astype(int)}
    m, ids = match(gt, dets)
    w = np.ones(len(ids)); w[3] = 2
    weighted = mean_ap([APCache(m[c]) for c in (1, 2, 3)], w)
    # duplicate image 4 (index 3) as a new image id
    dup = 1000
    gt2 = {**gt, "images": gt["images"] + [{"id": dup, "width": 640, "height": 480}],
           "annotations": gt["annotations"] + [{**a, "id": 10000 + a["id"], "image_id": dup}
                                               for a in gt["annotations"] if a["image_id"] == ids[3]]}
    extra = rows[rows[:, 0] == ids[3]].copy(); extra[:, 0] = dup
    rows2 = np.vstack([rows, extra])
    ref, _ = reference(gt2, rows2, "all")
    assert abs(weighted - ref) < 1e-9, (weighted, ref)
