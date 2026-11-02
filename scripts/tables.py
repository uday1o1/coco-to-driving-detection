"""Markdown tables for the README, generated from results/*.json.

    python scripts/tables.py > results/tables.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parents[1] / "results"
DS = [("coco_val", "COCO"), ("kitti_val", "KITTI"), ("nuimages_val", "nuImages")]


def load(n):
    return json.loads((RES / f"{n}.json").read_text())


def md(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def f3(x):
    return "-" if x is None or (isinstance(x, float) and (np.isnan(x) or x < 0)) else f"{x:.3f}"


def main_table():
    rows = load("main")
    models = []
    for r in rows:
        if r["model"] not in models:
            models.append(r["model"])
    get = {(r["model"], r["dataset"]): r for r in rows}
    out = []
    for m in models:
        line = [m]
        for ds, _ in DS:
            r = get.get((m, ds))
            line.append(f"{r['stats']['AP']:.3f} [{r['ci95']['AP'][0]:.3f}, {r['ci95']['AP'][1]:.3f}]" if r else "")
        out.append(line)
    return md(["model"] + [f"{n} mAP [95% CI]" for _, n in DS], out)


def class_table(ds):
    rows = [r for r in load("main") if r["dataset"] == ds]
    return md(["model", "AP", "AP50", "AP75", "vehicle", "person", "cyclist"],
              [[r["model"], f3(r["stats"]["AP"]), f3(r["stats"]["AP50"]), f3(r["stats"]["AP75"])]
               + [f3(r["per_class"][c]["AP"]) for c in ("vehicle", "person", "cyclist")] for r in rows])


def gate_table():
    return md(["model", "COCO val AP (this repo)", "published"],
              [[r["model"], f"{r['measured_AP']:.1f}", r["published_AP"] if r["published_AP"] else "n/a"]
               for r in load("gate")])


def reproduction_table():
    return md(["model", "images", "tutorial table", "ngdet rerun", "this repo, tutorial mode"],
              [[r["model"], f"{r['dataset']} ({r['images']})", f3(r["tutorial_table_AP"]), f"{r['ngdet_AP']:.4f}",
                f"{r['this_repo_tutorial_mode_AP']:.4f}" if "this_repo_tutorial_mode_AP" in r else "-"]
               for r in load("reproduction")["rows"]])


def audit_table():
    a = load("audit")
    out = []
    for ds, name in DS:
        rs = [r for r in a["summary"] if r["dataset"] == ds]
        if not rs:
            continue
        sh = {f: np.mean([r["shapley"][f] for r in rs]) for f in a["factors"]}
        out.append([name, f"{np.mean([r['tutorial_AP'] for r in rs]):.3f}",
                    f"{np.mean([r['corrected_AP'] for r in rs]):.3f}"] + [f"{sh[f]:+.3f}" for f in a["factors"]])
    return md(["dataset", "mean mAP, tutorial", "mean mAP, corrected"] + a["factors"], out)


def finetune_table():
    ft = load("finetune")
    out = []
    for name in ("Faster R-CNN v2", "YOLO11-M"):
        for ds, dn in DS:
            ap = [r["stats"]["AP"] for r in ft["runs"] if r["family"] == name and r["dataset"] == ds]
            d = [x for x in ft["diff_vs_zero_shot"] if x["model"] == name and x["dataset"] == ds]
            if not ap:
                continue
            out.append([name, dn, f"{np.mean(ap):.3f} ± {np.std(ap, ddof=1):.3f}",
                        f"{np.mean([x['diff'] for x in d]):+.3f}",
                        f"[{min(x['ci95'][0] for x in d):+.3f}, {max(x['ci95'][1] for x in d):+.3f}]"])
    return md(["model", "evaluated on", "fine-tuned mAP (3 seeds)", "change vs zero-shot",
               "range of per-seed 95% CIs"], out)


def latency_table():
    n, k = load("latency_nuimages"), load("latency_kitti")
    kk = {r["model"]: r for r in k["runs"]}
    return md(["model", "params (M)", "1600x900 fp32 ms", "1600x900 fp16 ms", "network only fp16 ms",
               "KITTI fp16 ms", "peak memory fp16 (MB)"],
              [[r["model"], r["params_M"], r["fp32_end_to_end_ms_median"], r["fp16_end_to_end_ms_median"],
                r["fp16_forward_ms_median"], kk[r["model"]]["fp16_end_to_end_ms_median"], r["fp16_peak_mem_MB"]]
               for r in n["runs"]])


def contrast_table():
    return md(["A", "B", "dataset", "AP(A) - AP(B)", "95% CI"],
              [[r["a"], r["b"], r["dataset"], f"{r['diff']:+.3f}", f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]"]
               for r in load("contrasts")])


def subgroup_table():
    out = []
    for r in load("subgroups")["rows"]:
        g = r["gap_vs_adult"]
        c, cs = g["human.pedestrian.child"], g["human.pedestrian.child/height_std"]
        w, ws = g["human.pedestrian.construction_worker"], g["human.pedestrian.construction_worker/height_std"]
        out.append([r["model"], f"{100 * c['gap']:+.1f} (p={c['p_holm']:.2f})", f"{100 * cs['gap']:+.1f}",
                    f"{100 * w['gap']:+.1f} (p={w['p_holm']:.2f})", f"{100 * ws['gap']:+.1f}"])
    return md(["model", "child gap, pts (Holm p)", "child gap at matched height",
               "construction-worker gap (Holm p)", "worker gap at matched height"], out)


if __name__ == "__main__":
    for title, fn in [("Main results", main_table), ("COCO check", gate_table),
                      ("Tutorial reproduction", reproduction_table), ("Protocol audit (mean Shapley)", audit_table),
                      ("KITTI val per class", lambda: class_table("kitti_val")),
                      ("nuImages val per class", lambda: class_table("nuimages_val")),
                      ("Fine-tuning", finetune_table), ("Latency", latency_table),
                      ("Paired contrasts", contrast_table), ("Pedestrian subgroups", subgroup_table)]:
        print(f"### {title}\n\n{fn()}\n")
