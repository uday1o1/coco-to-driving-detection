# ngdet evaluation report

Unified taxonomy: **driving3** = `['vehicle', 'person', 'cyclist']`

## Summary matrix — mAP @[.5:.95]

| model \ dataset | nuimages |
|---|---|
| torchvision:fasterrcnn_resnet50_fpn_v2 | 0.299 |
| yolo:yolo11x.pt | 0.249 |
| gdino:IDEA-Research/grounding-dino-tiny | 0.303 |

## Summary matrix — AP50

| model \ dataset | nuimages |
|---|---|
| torchvision:fasterrcnn_resnet50_fpn_v2 | 0.520 |
| yolo:yolo11x.pt | 0.375 |
| gdino:IDEA-Research/grounding-dino-tiny | 0.501 |

## Full COCO tables (per run)

### torchvision:fasterrcnn_resnet50_fpn_v2  ×  nuimages  (500 imgs, 84.7s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.299 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.520 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.289 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.138 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.279 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.453 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.182 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.385 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.399 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.225 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.382 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.561 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.417 |
| person | 0.257 |
| cyclist | 0.222 |

### yolo:yolo11x.pt  ×  nuimages  (500 imgs, 31.7s, open_vocab=False)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.249 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.375 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.257 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.063 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.218 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.491 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.154 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.300 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.303 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.078 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.271 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.591 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.400 |
| person | 0.170 |
| cyclist | 0.176 |

### gdino:IDEA-Research/grounding-dino-tiny  ×  nuimages  (500 imgs, 476.2s, open_vocab=True)

| Metric | Value |
|---|---|
| AP  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.303 |
| AP  @[IoU=0.50      | area=   all | maxDets=100] | 0.501 |
| AP  @[IoU=0.75      | area=   all | maxDets=100] | 0.311 |
| AP  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.123 |
| AP  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.286 |
| AP  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.487 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=  1] | 0.209 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets= 10] | 0.433 |
| AR  @[IoU=0.50:0.95 | area=   all | maxDets=100] | 0.454 |
| AR  @[IoU=0.50:0.95 | area= small | maxDets=100] | 0.300 |
| AR  @[IoU=0.50:0.95 | area=medium | maxDets=100] | 0.442 |
| AR  @[IoU=0.50:0.95 | area= large | maxDets=100] | 0.632 |

| Class | AP@[.5:.95] |
|---|---|
| vehicle | 0.349 |
| person | 0.287 |
| cyclist | 0.272 |
