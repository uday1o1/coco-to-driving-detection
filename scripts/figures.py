"""Figures for the README and report, from results/*.json.

    python scripts/figures.py            # writes figures/*.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RES, FIG = REPO / "results", REPO / "figures"
DS_LABEL = {"coco_val": "COCO val2017", "kitti_val": "KITTI val", "nuimages_val": "nuImages val"}
FAMILY_COLOR = {"two-stage CNN": "#1f77b4", "one-stage CNN": "#2ca02c", "YOLO": "#ff7f0e",
                "real-time DETR": "#d62728", "open vocabulary": "#9467bd"}
FACTOR_LABEL = {"semantics": "label semantics", "ignore": "ignore regions", "minsize": "minimum-size rule",
                "empty": "keep empty images", "threshold": "no score cut-off", "fusion": "rider fusion"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "figure.dpi": 200, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})


def load(name):
    p = RES / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def fig_protocol(audit):
    """Tutorial vs corrected mAP per model, and the Shapley split of the change."""
    rows = [r for r in audit["summary"] if r["dataset"] in ("kitti_val", "nuimages_val")]
    dsets = [d for d in ("kitti_val", "nuimages_val") if any(r["dataset"] == d for r in rows)]
    fig, axes = plt.subplots(1, 2 * len(dsets), figsize=(3.6 * len(dsets) * 2, 3.1),
                             gridspec_kw={"width_ratios": [1.1, 1] * len(dsets)})
    for k, ds in enumerate(dsets):
        rs = [r for r in rows if r["dataset"] == ds]
        rs.sort(key=lambda r: r["corrected_AP"])
        ax = axes[2 * k]
        y = np.arange(len(rs))
        for i, r in enumerate(rs):
            ax.plot([r["tutorial_AP"], r["corrected_AP"]], [i, i], color="#999", lw=1)
        ax.scatter([r["tutorial_AP"] for r in rs], y, s=14, color="#bbb", label="tutorial protocol", zorder=3)
        ax.scatter([r["corrected_AP"] for r in rs], y, s=14,
                   color=[FAMILY_COLOR[r["family"]] for r in rs], label="corrected protocol", zorder=3)
        ax.set_yticks(y, [r["model"] for r in rs])
        ax.set_xlabel("mAP@[.5:.95]")
        ax.set_title(f"{DS_LABEL[ds]}: mAP under each protocol")
        ax.legend(frameon=False, loc="lower right", fontsize=7)
        ax = axes[2 * k + 1]
        factors = audit["factors"]
        left_pos, left_neg = np.zeros(len(rs)), np.zeros(len(rs))
        cmap = plt.get_cmap("tab10")
        for j, f in enumerate(factors):
            v = np.array([r["shapley"][f] for r in rs])
            base = np.where(v >= 0, left_pos, left_neg)
            ax.barh(y, v, left=base, color=cmap(j), label=FACTOR_LABEL[f], height=0.7)
            left_pos += np.clip(v, 0, None); left_neg += np.clip(v, None, 0)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_yticks(y, [])
        ax.set_xlabel("contribution to the mAP change (Shapley)")
        ax.set_title("what moved the number")
        if k == len(dsets) - 1:
            ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "protocol_audit.png")
    plt.close(fig)


def fig_cross_dataset(main):
    """Per-class AP on each dataset for the main models, with clustered CIs."""
    models = [r["model"] for r in main if r["dataset"] == "coco_val" and r["spec"].count("@") == 0]
    classes = ["vehicle", "person", "cyclist"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, cls in zip(axes, classes):
        for j, ds in enumerate(("coco_val", "kitti_val", "nuimages_val")):
            vals, lo, hi = [], [], []
            for mname in models:
                r = next((x for x in main if x["model"] == mname and x["dataset"] == ds), None)
                v = r["per_class"][cls]["AP"] if r else np.nan
                c = r["ci95"].get(f"AP_{cls}", [np.nan, np.nan]) if r else [np.nan, np.nan]
                vals.append(v if v >= 0 else np.nan); lo.append(c[0]); hi.append(c[1])
            x = np.arange(len(models)) + (j - 1) * 0.27
            vals, lo, hi = map(np.array, (vals, lo, hi))
            ax.bar(x, vals, width=0.26, label=DS_LABEL[ds], color=["#888", "#4c72b0", "#dd8452"][j])
            ax.errorbar(x, vals, yerr=[vals - lo, hi - vals], fmt="none", ecolor="k", elinewidth=0.6, capsize=1.2)
        ax.set_xticks(np.arange(len(models)), models, rotation=75, ha="right")
        ax.set_title(f"{cls}")
    axes[0].set_ylabel("AP@[.5:.95]")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "cross_dataset_per_class.png")
    plt.close(fig)


def fig_height(main):
    """AP by object height on the driving sets, averaged over closed-set models per family."""
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.8), sharey=True)
    bins = ["0-40", "40-80", "80-160", "160-inf"]
    for ax, ds in zip(axes, ("kitti_val", "nuimages_val")):
        for fam, col in FAMILY_COLOR.items():
            rs = [r for r in main if r["dataset"] == ds and r["family"] == fam and "@" not in r["spec"]]
            if not rs:
                continue
            for cls, ls in (("vehicle", "-"), ("person", "--")):
                ys = [np.mean([r["height_bins"][b][cls] for r in rs if r["height_bins"][b][cls] >= 0]) for b in bins]
                ax.plot(range(4), ys, ls, color=col, marker="o", ms=3, label=f"{fam}, {cls}")
        ax.set_xticks(range(4), ["<40", "40-80", "80-160", ">160"])
        ax.set_xlabel("object height (px, original image)")
        ax.set_title(DS_LABEL[ds])
    axes[0].set_ylabel("AP@[.5:.95]")
    axes[1].legend(frameon=False, fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "ap_by_height.png")
    plt.close(fig)


def fig_speed(main, latency):
    """Accuracy against batch-1 fp16 end-to-end latency, one panel per dataset."""
    lat = {r["model"]: r for r in latency["runs"]}
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for ax, ds in zip(axes, ("coco_val", "kitti_val", "nuimages_val")):
        pts = []
        for r in main:
            if r["dataset"] != ds or r["spec"] not in lat:
                continue
            t = lat[r["spec"]]["fp16_end_to_end_ms_median"]
            pts.append((t, r["stats"]["AP"], r["model"], r["family"]))
            ax.scatter(t, r["stats"]["AP"], color=FAMILY_COLOR[r["family"]], s=16)
            ax.annotate(r["model"], (t, r["stats"]["AP"]), fontsize=5.5, xytext=(2, 2), textcoords="offset points")
        pts.sort()
        front, best = [], -1
        for t, ap, *_ in pts:
            if ap > best:
                front.append((t, ap)); best = ap
        if front:
            ax.plot(*zip(*front), color="k", lw=0.7, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("latency, ms per image (batch 1, fp16, end to end)")
        ax.set_title(DS_LABEL[ds])
    axes[0].set_ylabel("mAP@[.5:.95]")
    fig.tight_layout()
    fig.savefig(FIG / "speed_accuracy.png")
    plt.close(fig)


def fig_finetune(ft, main):
    """Zero-shot vs fine-tuned (three seeds) on each dataset."""
    zs = {("YOLO11-M", r["dataset"]): r["stats"]["AP"] for r in main if r["spec"] == "yolo:yolo11m.pt"}
    zs.update({("Faster R-CNN v2", r["dataset"]): r["stats"]["AP"] for r in main
               if r["spec"] == "tv:fasterrcnn_resnet50_fpn_v2"})
    fig, axes = plt.subplots(1, 3, figsize=(8, 2.6))
    for ax, ds in zip(axes, ("kitti_val", "nuimages_val", "coco_val")):
        for i, name in enumerate(("Faster R-CNN v2", "YOLO11-M")):
            seeds = [r["stats"]["AP"] for r in ft["runs"] if r["family"] == name and r["dataset"] == ds]
            ax.bar(i - 0.18, zs.get((name, ds), np.nan), width=0.34, color="#bbb", label="zero-shot" if i == 0 else None)
            ax.bar(i + 0.18, np.mean(seeds) if seeds else np.nan, width=0.34, color="#4c72b0",
                   label="fine-tuned on KITTI" if i == 0 else None)
            ax.scatter([i + 0.18] * len(seeds), seeds, color="k", s=6, zorder=3)
        ax.set_xticks([0, 1], ["Faster R-CNN v2", "YOLO11-M"])
        ax.set_title(DS_LABEL[ds])
    axes[0].set_ylabel("mAP@[.5:.95]")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "finetune.png")
    plt.close(fig)


def fig_subgroups(sg):
    groups = ["human.pedestrian.adult", "human.pedestrian.child", "human.pedestrian.construction_worker",
              "human.pedestrian.police_officer"]
    labels = ["adult", "child", "construction\nworker", "police\nofficer"]
    rows = [r for r in sg["rows"] if "seed" not in r["model"]]
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    w = 0.8 / len(rows)
    for i, r in enumerate(rows):
        x = np.arange(len(groups)) + (i - len(rows) / 2) * w
        v = np.array([r["recall"][g] for g in groups])
        lo = np.array([r["ci95"][g][0] for g in groups]); hi = np.array([r["ci95"][g][1] for g in groups])
        ax.bar(x, v, width=w, label=r["model"])
        ax.errorbar(x, v, yerr=[v - lo, hi - v], fmt="none", ecolor="k", elinewidth=0.4)
    ax.set_xticks(range(len(groups)), labels)
    ax.set_ylabel("recall at IoU 0.5\n(F1-optimal threshold)")
    ax.legend(frameon=False, fontsize=5.5, ncol=4, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "pedestrian_subgroups.png")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    audit, main_rows = load("audit"), load("main")
    if audit:
        fig_protocol(audit)
    if main_rows:
        fig_cross_dataset(main_rows)
        fig_height(main_rows)
    lat, ft, sg = load("latency_nuimages"), load("finetune"), load("subgroups")
    if main_rows and lat:
        fig_speed(main_rows, lat)
    if main_rows and ft:
        fig_finetune(ft, main_rows)
    if sg:
        fig_subgroups(sg)
    print("figures in", FIG)


if __name__ == "__main__":
    main()
