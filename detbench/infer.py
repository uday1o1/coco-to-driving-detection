"""Run one detector over one dataset and store every detection.

    python -m detbench.infer --model yolo:yolo11m.pt --native data/native/kitti_val.json \
        --images $DATA/kitti --out runs/dets/yolo11m__kitti_val.npz

Detections are stored in the model's own label vocabulary at score >= 0.001,
up to 300 per image, so any protocol can be applied afterwards. A JSON file
next to the .npz records the command, versions, hardware and timing.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .detectors import build


class _Images(torch.utils.data.Dataset):
    def __init__(self, records, root):
        self.records, self.root = records, Path(root)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        return r["id"], Image.open(self.root / r["file_name"]).convert("RGB")


def _collate(batch):
    return list(zip(*batch))   # module level so worker processes can pickle it (spawn on macOS)


def environment() -> dict:
    import torchvision, transformers, ultralytics
    env = {"python": platform.python_version(), "torch": torch.__version__,
           "torchvision": torchvision.__version__, "transformers": transformers.__version__,
           "ultralytics": ultralytics.__version__, "host": platform.node()}
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
        env["cuda"] = torch.version.cuda
    return env


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--native", type=Path, required=True, help="native ground-truth record")
    p.add_argument("--images", type=Path, required=True, help="dataset image root")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="first N images only (smoke tests)")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    records = json.loads(a.native.read_text())["images"]
    if a.limit:
        records = records[: a.limit]
    det = build(a.model, a.device)
    loader = torch.utils.data.DataLoader(_Images(records, a.images), batch_size=a.batch,
                                         num_workers=a.workers, collate_fn=_collate)
    ids, boxes, scores, labels = [], [], [], []
    t0, n = time.time(), 0
    for img_ids, ims in loader:
        for iid, (b, s, l) in zip(img_ids, det(list(ims), fp16=a.fp16)):
            ids.append(np.full(len(s), iid, np.int64))
            boxes.append(b.reshape(-1, 4).astype(np.float32))
            scores.append(s.astype(np.float32))
            labels.append(l.astype(np.int16))
        n += len(ims)
        if n % (a.batch * 100) < a.batch:
            print(f"{n}/{len(records)} images, {n / (time.time() - t0):.1f} img/s", flush=True)
    elapsed = time.time() - t0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, image_id=np.concatenate(ids), boxes=np.concatenate(boxes),
                        scores=np.concatenate(scores), labels=np.concatenate(labels),
                        names=np.array(det.names))
    meta = {"model": a.model, "native": str(a.native), "images": len(records),
            "seconds": round(elapsed, 1), "img_per_s": round(len(records) / elapsed, 2),
            "batch": a.batch, "fp16": a.fp16, "command": " ".join(sys.argv),
            "time": time.strftime("%Y-%m-%d %H:%M:%S %Z"), **environment()}
    a.out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
