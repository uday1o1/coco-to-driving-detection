# ngdet evaluation report

Unified taxonomy: **driving3** = `['vehicle', 'person', 'cyclist']`

## Summary matrix — mAP @[.5:.95]

| model \ dataset | kitti | coco |
|---|---|---|
| torchvision:fasterrcnn_resnet50_fpn_v2 | 0.236 | 0.581 |
| yolo:yolo11x.pt | 0.235 | 0.573 |
| gdino:IDEA-Research/grounding-dino-tiny | 0.186 | 0.531 |

## Summary matrix — AP50

| model \ dataset | kitti | coco |
|---|---|---|
| torchvision:fasterrcnn_resnet50_fpn_v2 | 0.460 | 0.781 |
| yolo:yolo11x.pt | 0.430 | 0.695 |
| gdino:IDEA-Research/grounding-dino-tiny | 0.365 | 0.696 |

## Generalization gap (COCO mAP − driving-dataset mAP)

Positive = drop when moving from the COCO training domain to a driving domain (higher = worse generalization).

| model | kitti |
|---|---|
| torchvision:fasterrcnn_resnet50_fpn_v2 | +0.344 |
| yolo:yolo11x.pt | +0.338 |
| gdino:IDEA-Research/grounding-dino-tiny | +0.345 |

## Full COCO tables (per run)

### torchvision:fasterrcnn_resnet50_fpn_v2  ×  kitti  (120 imgs, 21.9s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.236 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.460 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.204 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.139 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.231 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.332 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.083 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.292 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.314 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.290 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.303 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.395 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.463 |
| person | 0.246 |
| cyclist | 0.000 |

### torchvision:fasterrcnn_resnet50_fpn_v2  ×  coco  (120 imgs, 21.7s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.581 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.781 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.644 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.360 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.564 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.797 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.346 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.636 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.659 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.487 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.645 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.836 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.567 |
| person | 0.539 |
| cyclist | 0.637 |

### yolo:yolo11x.pt  ×  kitti  (120 imgs, 16.5s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.235 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.430 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.222 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.140 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.222 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.348 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.092 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.283 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.289 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.215 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.283 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.389 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.468 |
| person | 0.237 |
| cyclist | 0.000 |

### yolo:yolo11x.pt  ×  coco  (120 imgs, 16.4s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.573 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.695 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.631 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.264 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.591 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.855 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.299 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.604 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.609 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.320 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.622 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.877 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.580 |
| person | 0.547 |
| cyclist | 0.591 |

### gdino:IDEA-Research/grounding-dino-tiny  ×  kitti  (120 imgs, 98.8s, open_vocab=True)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.186 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.365 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.174 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.129 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.187 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.312 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.086 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.289 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.312 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.286 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.298 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.403 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.343 |
| person | 0.211 |
| cyclist | 0.004 |

### gdino:IDEA-Research/grounding-dino-tiny  ×  coco  (120 imgs, 108.9s, open_vocab=True)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.531 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.696 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.609 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.288 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.503 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.730 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.299 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.617 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.642 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.445 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.604 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.793 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.373 |
| person | 0.528 |
| cyclist | 0.692 |
