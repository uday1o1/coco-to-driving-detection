# ngdet evaluation report

Unified taxonomy: **driving3** = `['vehicle', 'person', 'cyclist']`

## Summary matrix — mAP @[.5:.95]

| model \ dataset | kitti |
|---|---|
| torchvision:fasterrcnn_resnet50_fpn_v2 | 0.279 |
| yolo:yolo11x.pt | 0.261 |
| gdino:IDEA-Research/grounding-dino-tiny | 0.226 |

## Summary matrix — AP50

| model \ dataset | kitti |
|---|---|
| torchvision:fasterrcnn_resnet50_fpn_v2 | 0.467 |
| yolo:yolo11x.pt | 0.435 |
| gdino:IDEA-Research/grounding-dino-tiny | 0.391 |

## Full COCO tables (per run)

### torchvision:fasterrcnn_resnet50_fpn_v2  ×  kitti  (120 imgs, 21.8s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.279 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.467 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.316 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.179 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.298 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.357 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.141 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.363 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.378 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.345 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.382 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.418 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.441 |
| person | 0.396 |
| cyclist | 0.001 |

### yolo:yolo11x.pt  ×  kitti  (120 imgs, 14.2s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.261 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.435 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.292 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.161 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.284 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.346 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.129 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.337 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.339 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.252 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.353 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.411 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.454 |
| person | 0.328 |
| cyclist | 0.000 |

### gdino:IDEA-Research/grounding-dino-tiny  ×  kitti  (120 imgs, 115.7s, open_vocab=True)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.226 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.391 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.249 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.180 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.234 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.306 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.113 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.421 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.440 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.448 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.402 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.513 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.349 |
| person | 0.303 |
| cyclist | 0.026 |
