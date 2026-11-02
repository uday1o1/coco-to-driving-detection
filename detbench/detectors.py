"""Detector wrappers with one interface: a batch of PIL images in, native boxes out.

Each wrapper keeps its model's official preprocessing and returns, per image,
(boxes xyxy in original pixels, scores, label indices into `self.names`).
Labels stay in the model's own vocabulary; mapping to the driving taxonomy
happens at evaluation time so one inference run serves every protocol.

Model specs:
  tv:<torchvision builder>        e.g. tv:fasterrcnn_resnet50_fpn_v2
  yolo:<weights>[@<imgsz>|single]  e.g. yolo:yolo11m.pt, yolo:yolo11m.pt@1280
  dfine:<n|m|x>[@letterbox]       official D-FINE, e.g. dfine:m
  hfdfine:<hf id>[@letterbox]     the Hugging Face port, kept for the comparison
  gdino:<hf id>[@<prompt set>]    Grounding DINO family (GDINO, MM-GDINO, LLMDet)
  <spec>#<checkpoint>             fine-tuned weights for tv: and yolo: specs
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from .taxonomy import PROMPTS

SCORE_MIN = 0.001   # keep everything a COCO-style evaluation could use
MAX_DETS = 300      # per image, before the evaluator's 100-per-class cap


def _np(t):
    return t.detach().float().cpu().numpy()


class TorchvisionDetector:
    """Faster R-CNN / RetinaNet / FCOS from torchvision with COCO weights.

    Preprocessing is torchvision's GeneralizedRCNNTransform: resize so the
    short side is 800 px, capped at 1333 px on the long side. Class-wise NMS
    at IoU 0.5 (torchvision default).
    """

    def __init__(self, name, device, checkpoint=None, num_classes=None):
        import torchvision.models.detection as D
        builder = getattr(D, name)
        if checkpoint:
            self.model = builder(weights=None, weights_backbone=None, num_classes=num_classes)
            self.model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
            self.names = ["__background__", "vehicle", "person", "cyclist"][:num_classes]
        else:
            weights = D.__dict__[_WEIGHTS[name]].DEFAULT
            self.model = builder(weights=weights)
            self.names = list(weights.meta["categories"])
        m = self.model
        if hasattr(m, "roi_heads"):
            m.roi_heads.score_thresh, m.roi_heads.detections_per_img = SCORE_MIN, MAX_DETS
        else:
            m.score_thresh, m.detections_per_img = SCORE_MIN, MAX_DETS
        self.model.eval().to(device)
        self.device = device

    @torch.inference_mode()
    def __call__(self, images, fp16=False):
        import torchvision.transforms.functional as F
        x = [F.to_tensor(im).to(self.device) for im in images]
        with torch.autocast("cuda", enabled=fp16 and self.device.startswith("cuda")):
            out = self.model(x)
        return [(_np(o["boxes"]), _np(o["scores"]), _np(o["labels"]).astype(np.int64)) for o in out]


_WEIGHTS = {
    "fasterrcnn_resnet50_fpn": "FasterRCNN_ResNet50_FPN_Weights",
    "fasterrcnn_resnet50_fpn_v2": "FasterRCNN_ResNet50_FPN_V2_Weights",
    "retinanet_resnet50_fpn": "RetinaNet_ResNet50_FPN_Weights",
    "retinanet_resnet50_fpn_v2": "RetinaNet_ResNet50_FPN_V2_Weights",
    "fcos_resnet50_fpn": "FCOS_ResNet50_FPN_Weights",
}


class YoloDetector:
    """Ultralytics YOLO. Letterbox to `imgsz` (aspect kept), NMS at IoU 0.7.

    By default NMS is multi-label, as in Ultralytics' own COCO validation
    (models/yolo/detect/val.py): a box may be kept for several classes. Its
    prediction mode keeps one label per box, which costs about 0.8 COCO AP;
    every other detector here also scores each box per class. `single_label`
    restores prediction mode, used only to match the ngdet harness.
    """

    def __init__(self, weights, device, imgsz=640, single_label=False):
        from ultralytics import YOLO
        import ultralytics.utils.nms as nms
        if not hasattr(nms, "_nms_single"):
            nms._nms_single = nms.non_max_suppression
        if single_label:
            nms.non_max_suppression = nms._nms_single
        else:
            def multi(*a, **k):
                k.setdefault("multi_label", True)
                return nms._nms_single(*a, **k)
            nms.non_max_suppression = multi
        self.model = YOLO(weights)
        self.names = [self.model.names[i] for i in range(len(self.model.names))]
        self.device, self.imgsz = device, imgsz

    def __call__(self, images, fp16=False):
        res = self.model.predict(images, imgsz=self.imgsz, conf=SCORE_MIN, iou=0.7,
                                 max_det=MAX_DETS, half=fp16, device=self.device,
                                 verbose=False)
        return [(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                 r.boxes.cls.cpu().numpy().astype(np.int64)) for r in res]


class DFineDetector:
    """D-FINE through Hugging Face transformers.

    Official preprocessing resizes to 640x640 without keeping the aspect
    ratio. `letterbox=True` instead scales the long side to 640 and pads,
    the YOLO convention, to separate input distortion from architecture.
    Outputs the top 300 (query, class) pairs, no NMS.
    """

    def __init__(self, hf_id, device, letterbox=False):
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        self.proc = AutoImageProcessor.from_pretrained(hf_id)
        self.model = AutoModelForObjectDetection.from_pretrained(hf_id).eval().to(device)
        self.names = [self.model.config.id2label[i] for i in range(len(self.model.config.id2label))]
        self.device, self.letterbox = device, letterbox

    @torch.inference_mode()
    def __call__(self, images, fp16=False):
        sizes = [im.size for im in images]                       # (w, h)
        if self.letterbox:
            images, scales = zip(*[_letterbox(im, 640) for im in images])
        inputs = self.proc(images=list(images), return_tensors="pt").to(self.device)
        with torch.autocast("cuda", enabled=fp16 and self.device.startswith("cuda")):
            out = self.model(**inputs)
        logits, boxes = out.logits.float().sigmoid(), out.pred_boxes.float()
        results = []
        for i, (w, h) in enumerate(sizes):
            s, k = logits[i].flatten().topk(MAX_DETS)
            q, c = k // logits.shape[-1], k % logits.shape[-1]
            b = _cxcywh_to_xyxy(boxes[i, q])
            if self.letterbox:
                b = b * 640 / scales[i]
            else:
                b = b * torch.tensor([w, h, w, h], device=b.device)
            results.append((_np(b), _np(s), _np(c).astype(np.int64)))
        return results


class DFineOfficialDetector:
    """D-FINE from the authors' repository (Peterande/D-FINE) with their
    released COCO weights.

    The Hugging Face port (transformers 5.22.1, ustc-community checkpoints)
    scores about 4 AP lower on COCO val with identical preprocessing, so the
    official code is used. Needs the repository at $DFINE_SRC (default
    ~/dfine_src, set up by scripts/setup.sh). Preprocessing is the official
    one: resize to 640x640 without keeping aspect ratio, scale to [0, 1].
    The post-processor keeps the top 300 (query, class) pairs, no NMS.
    """
    COMMIT = "956d1709314c2c6a4df6f34de232054578a7449f"
    URL = "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_{}_coco.pth"

    def __init__(self, size, device, letterbox=False):
        import os
        import sys
        import urllib.request
        from pathlib import Path
        src = Path(os.environ.get("DFINE_SRC", Path.home() / "dfine_src"))
        sys.path.insert(0, str(src))
        from src.core import YAMLConfig
        w = src / "weights" / f"dfine_{size}_coco.pth"
        if not w.exists():
            w.parent.mkdir(exist_ok=True)
            urllib.request.urlretrieve(self.URL.format(size), w)
        cfg = YAMLConfig(str(src / f"configs/dfine/dfine_hgnetv2_{size}_coco.yml"))
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        state = torch.load(w, map_location="cpu")
        cfg.model.load_state_dict(state["ema"]["module"] if "ema" in state else state["model"])
        self.model = cfg.model.deploy().eval().to(device)
        self.post = cfg.postprocessor.deploy()
        import json
        self.names = json.loads((Path(__file__).with_name("coco80.json")).read_text())
        self.device, self.letterbox = device, letterbox

    @torch.inference_mode()
    def __call__(self, images, fp16=False):
        import torchvision.transforms.functional as F
        if self.letterbox:
            images, scales = zip(*[_letterbox(im, 640) for im in images])
            sizes = [(640, 640)] * len(images)
        else:
            sizes = [im.size for im in images]
        x = torch.stack([F.to_tensor(im.resize((640, 640), Image.BILINEAR)) for im in images]).to(self.device)
        with torch.autocast("cuda", enabled=fp16 and self.device.startswith("cuda")):
            labels, boxes, scores = self.post(self.model(x), torch.tensor(sizes, device=self.device))
        out = []
        for i in range(len(images)):
            b = boxes[i].float()
            if self.letterbox:
                b = b / scales[i]
            out.append((_np(b), _np(scores[i]), _np(labels[i]).astype(np.int64)))
        return out


class GroundingDinoDetector:
    """Grounding DINO, MM-Grounding-DINO and LLMDet through transformers.

    The prompt is the official caption format ("car . truck . bus ."). Each
    term's score is the mean sigmoid over its tokens, then the top 300
    (query, term) pairs are kept, as in Grounding DINO's own COCO evaluation
    (demo/test_ap_on_coco.py, PostProcessCocoGrounding).
    """

    def __init__(self, hf_id, device, prompt="synonyms", terms=None):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(hf_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(hf_id).eval().to(device)
        if terms is None:   # names are the COCO-style labels the terms stand for
            terms = list(PROMPTS[prompt])
            self.names = [PROMPTS[prompt][t] for t in terms]
        else:
            self.names = list(terms)
        self.caption, spans = _caption(terms)
        tok = self.proc.tokenizer(self.caption, return_offsets_mapping=True)
        L = len(tok["input_ids"])
        pos = torch.zeros(len(terms), L)
        for j, word_spans in enumerate(spans):
            for beg, end in word_spans:
                for t, (a, b) in enumerate(tok["offset_mapping"]):
                    if b > a and a < end and b > beg:
                        pos[j, t] = 1
        self.pos = (pos / pos.sum(1, keepdim=True).clamp(min=1e-6)).to(device)
        self.device = device

    @torch.inference_mode()
    def __call__(self, images, fp16=False):
        sizes = [im.size for im in images]
        inputs = self.proc(images=list(images), text=[self.caption] * len(images),
                           return_tensors="pt").to(self.device)
        with torch.autocast("cuda", enabled=fp16 and self.device.startswith("cuda")):
            out = self.model(**inputs)
        prob = out.logits.float().sigmoid()[..., : self.pos.shape[1]] @ self.pos.T  # B,Q,C
        results = []
        for i, (w, h) in enumerate(sizes):
            s, k = prob[i].flatten().topk(min(MAX_DETS, prob[i].numel()))
            q, c = k // prob.shape[-1], k % prob.shape[-1]
            b = _cxcywh_to_xyxy(out.pred_boxes[i, q].float()) * torch.tensor([w, h, w, h], device=prob.device)
            keep = s >= SCORE_MIN
            results.append((_np(b[keep]), _np(s[keep]), _np(c[keep]).astype(np.int64)))
        return results


def _caption(terms):
    """Grounding DINO caption and per-term character spans (one per word)."""
    caption, spans = "", []
    for term in terms:
        word_spans = []
        for word in term.lower().split():
            if caption:
                caption += " "
            word_spans.append((len(caption), len(caption) + len(word)))
            caption += word
        caption += " ."
        spans.append(word_spans)
    return caption, spans


def _cxcywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


def _letterbox(im, size):
    s = size / max(im.size)
    w, h = round(im.width * s), round(im.height * s)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    canvas.paste(im.resize((w, h), Image.BILINEAR), (0, 0))
    return canvas, s


def build(spec: str, device: str = "cuda"):
    spec, _, ckpt = spec.partition("#")
    kind, _, rest = spec.partition(":")
    name, _, opt = rest.partition("@")
    if kind == "tv":
        return TorchvisionDetector(name, device, checkpoint=ckpt or None,
                                   num_classes=4 if ckpt else None)
    if kind == "yolo":
        return YoloDetector(ckpt or name, device, imgsz=int(opt) if opt.isdigit() else 640,
                            single_label=opt == "single")
    if kind == "dfine":
        return DFineOfficialDetector(name, device, letterbox=opt == "letterbox")
    if kind == "hfdfine":
        return DFineDetector(name, device, letterbox=opt == "letterbox")
    if kind == "gdino":
        if opt == "coco80":
            import json
            from pathlib import Path
            names = json.loads((Path(__file__).with_name("coco80.json")).read_text())
            return GroundingDinoDetector(name, device, terms=names)
        return GroundingDinoDetector(name, device, prompt=opt or "synonyms")
    raise ValueError(f"unknown model spec {spec}")
