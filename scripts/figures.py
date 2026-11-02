"""Figures from results/*.json.

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
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    rows = [r for r in audit["summary"] if r["dataset"] in ("kitti_val", "nuimages_val")]
    dsets = [d for d in ("kitti_val", "nuimages_val") if any(r["dataset"] == d for r in rows)]
    factors = audit["factors"]
    cmap = plt.get_cmap("Set2")   # distinct from the model-family colours
    fig, axes = plt.subplots(len(dsets), 2, figsize=(7.2, 2.7 * len(dsets)),
                             gridspec_kw={"width_ratios": [1.05, 1]}, squeeze=False)
    for k, ds in enumerate(dsets):
        rs = sorted([r for r in rows if r["dataset"] == ds], key=lambda r: r["corrected_AP"])
        y = np.arange(len(rs))
        ax = axes[k, 0]
        for i, r in enumerate(rs):
            ax.plot([r["tutorial_AP"], r["corrected_AP"]], [i, i], color="#999", lw=1)
        ax.scatter([r["tutorial_AP"] for r in rs], y, s=14, color="#bbb", zorder=3)
        ax.scatter([r["corrected_AP"] for r in rs], y, s=14, color=[FAMILY_COLOR[r["family"]] for r in rs], zorder=3)
        ax.set_yticks(y, [r["model"] for r in rs], fontsize=6.5)
        ax.set_xlabel("mAP@[.5:.95]")
        ax.set_title(f"{DS_LABEL[ds]}: tutorial (grey) to corrected protocol")
        ax = axes[k, 1]
        pos, neg = np.zeros(len(rs)), np.zeros(len(rs))
        for j, f in enumerate(factors):
            v = np.array([r["shapley"][f] for r in rs])
            ax.barh(y, v, left=np.where(v >= 0, pos, neg), color=cmap(j), height=0.7)
            pos += np.clip(v, 0, None); neg += np.clip(v, None, 0)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_yticks(y, [])
        ax.set_xlabel("contribution to the change in mAP (Shapley value)")
        ax.tick_params(axis="x", labelsize=7)
        ax.set_title("which protocol choice moved it")
    handles = [Patch(color=cmap(j), label=FACTOR_LABEL[f]) for j, f in enumerate(factors)]
    handles += [Line2D([], [], marker="o", ls="", color=c, label=f) for f, c in FAMILY_COLOR.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=6.5,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
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
    axes[2].legend(frameon=False, fontsize=7, loc="upper left")
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
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=6, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIG / "ap_by_height.png")
    plt.close(fig)


LABEL_OFFSET = {("kitti_val", "Faster R-CNN v2"): (-14, 6), ("kitti_val", "YOLO11-M at 1280"): (3, -9),
                ("kitti_val", "Faster R-CNN"): (-50, -3), ("nuimages_val", "D-FINE-M letterboxed"): (-38, -6)}
LABEL_TEXT = {("nuimages_val", "D-FINE-M letterboxed"): "D-FINE-M\nletterboxed"}


def fig_speed(main, lat_nuim, lat_kitti):
    """Accuracy against batch-1 fp32 end-to-end latency (the precision the accuracy
    runs used), one panel per dataset.
    KITTI uses latency measured on KITTI frames; COCO and nuImages use latency
    on a 1600x900 nuImages frame, the cost of one driving camera frame."""
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, ds in zip(axes, ("coco_val", "kitti_val", "nuimages_val")):
        lat = {r["model"]: r for r in (lat_kitti if ds == "kitti_val" else lat_nuim)["runs"]}
        pts = []
        for r in main:
            if r["dataset"] != ds or r["spec"] not in lat:
                continue
            t = lat[r["spec"]]["fp32_end_to_end_ms_median"]
            pts.append((t, r["stats"]["AP"], r["model"], r["family"]))
            ax.scatter(t, r["stats"]["AP"], color=FAMILY_COLOR[r["family"]], s=16)
            off = LABEL_OFFSET.get((ds, r["model"]), (2, 2))   # move labels that would collide
            ax.annotate(LABEL_TEXT.get((ds, r["model"]), r["model"]), (t, r["stats"]["AP"]), fontsize=5.5, xytext=off,
                        textcoords="offset points")
        pts.sort()
        front, best = [], -1
        for t, ap, *_ in pts:
            if ap > best:
                front.append((t, ap)); best = ap
        if front:
            ax.plot(*zip(*front), color="k", lw=0.7, ls=":")
        ax.set_xscale("log")
        ax.set_xticks([5, 10, 20, 50, 100], ["5", "10", "20", "50", "100"])
        ax.minorticks_off()
        res = "KITTI frame" if ds == "kitti_val" else "1600x900 frame"
        ax.set_xlabel(f"ms per {res} (batch 1, fp32, end to end)")
        ax.set_title(DS_LABEL[ds])
    axes[0].set_ylabel("mAP@[.5:.95]")
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=c, label=f) for f, c in FAMILY_COLOR.items()]
               + [Line2D([], [], color="k", ls=":", lw=0.7, label="more accurate than every faster model")],
               frameon=False, fontsize=7, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
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
    axes[1].legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "finetune.png")
    plt.close(fig)


def fig_subgroups(sg):
    """Recall gap against adult pedestrians (adult minus group), raw and at
    matched height, with clustered 95% intervals."""
    rows = [r for r in sg["rows"] if "YOLO11-M fine-tuned" not in r["model"]]
    groups = [("human.pedestrian.child", "child"), ("human.pedestrian.construction_worker", "construction worker")]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, (g, label) in zip(axes, groups):
        for dy, key, mk, name in ((0.15, g, "o", "raw"), (-0.15, g + "/height_std", "s", "at matched height")):
            d = [r["gap_vs_adult"][key] for r in rows]
            v = 100 * np.array([x["gap"] for x in d])
            lo = 100 * np.array([x["ci95"][0] for x in d]); hi = 100 * np.array([x["ci95"][1] for x in d])
            ax.errorbar(v, y + dy, xerr=[v - lo, hi - v], fmt=mk, ms=3.5, lw=0.8, capsize=0,
                        color="#333" if mk == "o" else "#c44e52", mfc="white" if mk == "o" else "#c44e52",
                        label=name)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_title(f"adult recall minus {label} recall")
        ax.set_xlabel("percentage points (positive = group missed more)")
    axes[0].set_yticks(y, [r["model"].replace(" fine-tuned", " FT").replace("(seed ", "s").replace(")", "")
                           for r in rows], fontsize=6.5)
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
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
    lat, latk, ft, sg = load("latency_nuimages"), load("latency_kitti"), load("finetune"), load("subgroups")
    if main_rows and lat and latk:
        fig_speed(main_rows, lat, latk)
    if main_rows and ft:
        fig_finetune(ft, main_rows)
    if sg:
        fig_subgroups(sg)
    print("figures in", FIG)


if __name__ == "__main__":
    main()
