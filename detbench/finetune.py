"""Fine-tune a COCO-pretrained detector on KITTI train (Chen et al. split).

    python -m detbench.finetune yolo  --kitti $DATA/kitti --native $DATA/native/kitti_train.json --seed 0 --out $RUNS/ft
    python -m detbench.finetune frcnn --kitti $DATA/kitti --native $DATA/native/kitti_train.json --seed 0 --out $RUNS/ft

Training boxes are the scored classes of the corrected protocol (vehicle,
person, cyclist). DontCare, Misc and Person_sitting are left out, since
neither trainer supports ignore regions. There is no model selection: the
schedule is fixed and the last epoch is kept, so KITTI val, nuImages val and
COCO val all stay untouched until the final evaluation.

  yolo   Ultralytics defaults for YOLO11-M: 100 epochs, 640 px letterbox,
         batch 16, SGD/AdamW auto, mosaic closed for the last 10 epochs.
  frcnn  torchvision Faster R-CNN v2: new 4-way box predictor, 12 epochs,
         batch 8, SGD lr 0.01 momentum 0.9 weight decay 1e-4, linear warmup
         over 500 iterations, lr x0.1 after epochs 8 and 11, horizontal flip,
         mixed precision.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import taxonomy as tx


def kitti_targets(native: dict):
    """image record -> (boxes xyxy, unified class index 0..2) for training."""
    by = {im["id"]: [] for im in native["images"]}
    for b in native["boxes"]:
        cls = tx.CORRECTED["kitti"].get(b["fine"])
        if cls:
            by[b["image_id"]].append((b["xyxy"], tx.CID[cls]))
    return [(im, by[im["id"]]) for im in native["images"]]


def train_yolo(a):
    from ultralytics import YOLO
    root = a.out / "kitti_yolo"
    (root / "images/train").mkdir(parents=True, exist_ok=True)
    (root / "labels/train").mkdir(parents=True, exist_ok=True)
    for im, boxes in kitti_targets(json.loads(a.native.read_text())):
        src = a.kitti / im["file_name"]
        dst = root / "images/train" / src.name
        if not dst.exists():
            dst.symlink_to(src)
        W, H = Image.open(src).size
        lines = [f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}"
                 for (x1, y1, x2, y2), c in boxes]
        (root / "labels/train" / f"{src.stem}.txt").write_text("\n".join(lines))
    yaml = root / "data.yaml"
    yaml.write_text(f"path: {root}\ntrain: images/train\nval: images/train\n"
                    f"names: {dict(enumerate(tx.CLASSES))}\n")
    model = YOLO(a.weights or "yolo11m.pt")
    t0 = time.time()
    model.train(data=str(yaml), epochs=a.epochs or 100, imgsz=640, batch=16, seed=a.seed,
                deterministic=True, val=False, project=str(a.out), name=f"yolo11m_seed{a.seed}",
                exist_ok=True, workers=12, plots=False)
    (a.out / f"yolo11m_seed{a.seed}" / "done.json").write_text(
        json.dumps({"seconds": round(time.time() - t0), "epochs": a.epochs or 100, "seed": a.seed}))


def train_frcnn(a):
    import torchvision
    import torchvision.transforms.functional as F
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)
    model.roi_heads.box_predictor = FastRCNNPredictor(
        model.roi_heads.box_predictor.cls_score.in_features, len(tx.CLASSES) + 1)
    model.cuda().train()
    items = [(im, b) for im, b in kitti_targets(json.loads(a.native.read_text())) if b]

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(items)

        def __getitem__(self, i):
            im, boxes = items[i]
            img = F.to_tensor(Image.open(a.kitti / im["file_name"]).convert("RGB"))
            bx = torch.tensor([b for b, _ in boxes], dtype=torch.float32)
            lb = torch.tensor([c + 1 for _, c in boxes], dtype=torch.int64)
            if random.random() < 0.5:
                img = img.flip(-1)
                w = img.shape[-1]
                bx[:, [0, 2]] = w - bx[:, [2, 0]]
            return img, {"boxes": bx, "labels": lb}

    g = torch.Generator().manual_seed(a.seed)
    dl = torch.utils.data.DataLoader(DS(), batch_size=8, shuffle=True, num_workers=12,
                                     collate_fn=lambda b: tuple(zip(*b)), generator=g,
                                     worker_init_fn=lambda w: random.seed(a.seed * 100 + w))
    epochs = a.epochs or 12
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=1e-4)
    steps = epochs * len(dl)
    milestones = [8 * len(dl), 11 * len(dl)] if epochs == 12 else [int(steps * 2 / 3), int(steps * 11 / 12)]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 500) * 0.1 ** sum(s >= m for m in milestones))
    scaler = torch.amp.GradScaler()
    out = a.out / f"frcnn_v2_seed{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train_log.txt", "w")
    t0, step = time.time(), 0
    for ep in range(epochs):
        for imgs, tgts in dl:
            imgs = [i.cuda(non_blocking=True) for i in imgs]
            tgts = [{k: v.cuda() for k, v in t.items()} for t in tgts]
            with torch.autocast("cuda"):
                loss = sum(model(imgs, tgts).values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step(); step += 1
            if step % 50 == 0:
                msg = f"epoch {ep} step {step}/{steps} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.5f} {time.time() - t0:.0f}s"
                print(msg, flush=True); log.write(msg + "\n"); log.flush()
    torch.save(model.state_dict(), out / "last.pt")
    (out / "done.json").write_text(json.dumps({"seconds": round(time.time() - t0), "steps": steps,
                                               "epochs": epochs, "seed": a.seed}))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("trainer", choices=["yolo", "frcnn"])
    p.add_argument("--kitti", type=Path, required=True)
    p.add_argument("--native", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=0, help="0 = the default schedule")
    p.add_argument("--weights", default=None)
    a = p.parse_args()
    (train_yolo if a.trainer == "yolo" else train_frcnn)(a)


if __name__ == "__main__":
    main()
