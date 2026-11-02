"""Batch-1 latency, peak GPU memory and parameter count per detector.

Two timings per image, both synchronised with the GPU:
  end_to_end  the wrapper call on an already decoded PIL image: the library's
              own preprocessing (resize, normalise, letterbox), the forward
              pass and post-processing (NMS or top-k), as deployed.
  forward     the network alone on a prepared input already on the GPU, so a
              library's CPU preprocessing is not charged to the architecture.
              For torchvision this still includes its on-GPU resize and NMS,
              which are inside the module; for YOLO it excludes NMS.
Reported as median and 90th percentile over `--iters` images after
`--warmup` images, in fp32 and fp16 (autocast; half weights for YOLO).

    python -m detbench.latency --models yolo:yolo11m.pt tv:fcos_resnet50_fpn \
        --native $DATA/native/nuimages_val.json --images $DATA/nuimages --out latency.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .detectors import (DFineDetector, GroundingDinoDetector, TorchvisionDetector,
                        YoloDetector, _letterbox, build)
from .infer import environment


def _sync_time(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) * 1000


def forward_fn(det, im, fp16):
    """A zero-argument callable running only the network on a prepared input."""
    ac = lambda: torch.autocast("cuda", enabled=fp16)
    if isinstance(det, TorchvisionDetector):
        import torchvision.transforms.functional as F
        x = [F.to_tensor(im).cuda()]
        def f():
            with torch.inference_mode(), ac():
                det.model(x)
        return f
    if isinstance(det, YoloDetector):
        det([im], fp16=fp16)                            # sets up the predictor
        p = det.model.predictor
        x = p.preprocess([np.asarray(im)[:, :, ::-1]])
        def f():
            with torch.inference_mode():
                p.inference(x)
        return f
    if isinstance(det, DFineDetector):
        src = _letterbox(im, 640)[0] if det.letterbox else im
        inputs = det.proc(images=[src], return_tensors="pt").to("cuda")
    elif isinstance(det, GroundingDinoDetector):
        inputs = det.proc(images=[im], text=[det.caption], return_tensors="pt").to("cuda")
    else:
        raise TypeError(type(det))
    def f():
        with torch.inference_mode(), ac():
            det.model(**inputs)
    return f


def bench(spec, images, warmup, iters):
    det = build(spec, "cuda")
    net = det.model.model if isinstance(det, YoloDetector) else det.model
    out = {"model": spec, "params_M": round(sum(p.numel() for p in net.parameters()) / 1e6, 2)}
    for fp16 in (False, True):
        key = "fp16" if fp16 else "fp32"
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        e2e = [_sync_time(lambda: det([im], fp16=fp16)) for im in images[: warmup + iters]][warmup:]
        out[f"{key}_peak_mem_MB"] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
        fwd = []
        for im in images[: warmup + iters]:
            f = forward_fn(det, im, fp16)
            fwd.append(_sync_time(f))
        fwd = fwd[warmup:]
        for name, xs in (("end_to_end", e2e), ("forward", fwd)):
            out[f"{key}_{name}_ms_median"] = round(float(np.median(xs)), 2)
            out[f"{key}_{name}_ms_p90"] = round(float(np.percentile(xs, 90)), 2)
        out[f"{key}_fps"] = round(1000 / out[f"{key}_end_to_end_ms_median"], 1)
    del det
    torch.cuda.empty_cache()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--native", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=200)
    a = p.parse_args()
    recs = json.loads(a.native.read_text())["images"][: a.warmup + a.iters]
    images = [Image.open(a.images / r["file_name"]).convert("RGB") for r in recs]
    results = json.loads(a.out.read_text()) if a.out.exists() else {"runs": []}
    done = {r["model"] for r in results["runs"]}
    results["env"], results["input"] = environment(), str(a.native)
    for spec in a.models:
        if spec in done:
            continue
        r = bench(spec, images, a.warmup, a.iters)
        print(json.dumps(r), flush=True)
        results["runs"].append(r)
        a.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
