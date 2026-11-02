### Main results

| model | COCO mAP [95% CI] | KITTI mAP [95% CI] | nuImages mAP [95% CI] |
|---|---|---|---|
| Faster R-CNN | 0.490 [0.480, 0.500] | 0.256 [0.236, 0.314] | 0.313 [0.291, 0.331] |
| RetinaNet | 0.473 [0.463, 0.484] | 0.259 [0.240, 0.308] | 0.297 [0.275, 0.317] |
| FCOS | 0.507 [0.496, 0.518] | 0.245 [0.227, 0.281] | 0.282 [0.258, 0.299] |
| Faster R-CNN v2 | 0.577 [0.569, 0.587] | 0.286 [0.261, 0.364] | 0.391 [0.366, 0.409] |
| YOLO11-N | 0.494 [0.482, 0.506] | 0.224 [0.210, 0.259] | 0.221 [0.202, 0.236] |
| YOLO11-M | 0.609 [0.599, 0.620] | 0.268 [0.249, 0.326] | 0.333 [0.306, 0.352] |
| YOLO11-X | 0.636 [0.625, 0.647] | 0.272 [0.253, 0.334] | 0.367 [0.339, 0.391] |
| D-FINE-N | 0.498 [0.486, 0.509] | 0.232 [0.218, 0.269] | 0.248 [0.225, 0.267] |
| D-FINE-M | 0.598 [0.587, 0.609] | 0.276 [0.253, 0.329] | 0.326 [0.302, 0.345] |
| D-FINE-X | 0.620 [0.609, 0.631] | 0.283 [0.260, 0.342] | 0.343 [0.318, 0.361] |
| Grounding DINO-T | 0.570 [0.559, 0.582] | 0.313 [0.287, 0.370] | 0.390 [0.359, 0.410] |
| MM-GDINO-T | 0.578 [0.567, 0.590] | 0.269 [0.250, 0.322] | 0.350 [0.333, 0.372] |
| LLMDet-T | 0.585 [0.574, 0.596] | 0.287 [0.264, 0.350] | 0.378 [0.358, 0.396] |
| YOLO11-M at 1280 |  | 0.286 [0.261, 0.358] | 0.404 [0.380, 0.426] |
| D-FINE-M letterboxed |  | 0.263 [0.246, 0.310] | 0.289 [0.265, 0.308] |

### COCO check

| model | COCO val AP (this repo) | published |
|---|---|---|
| Faster R-CNN | 37.2 | 37.0 |
| RetinaNet | 35.9 | 36.4 |
| FCOS | 39.1 | 39.2 |
| Faster R-CNN v2 | 47.3 | 46.7 |
| YOLO11-N | 39.3 | 39.5 |
| YOLO11-M | 51.3 | 51.5 |
| YOLO11-X | 54.5 | 54.7 |
| D-FINE-N | 42.8 | 42.8 |
| D-FINE-M | 52.3 | 52.3 |
| D-FINE-X | 55.8 | 55.8 |
| Grounding DINO-T | 48.5 | 48.4 |
| MM-GDINO-T | 50.4 | 50.4 |
| LLMDet-T | 55.5 | n/a |
| D-FINE-N (Hugging Face port) | 42.8 | 42.8 |
| D-FINE-M (Hugging Face port) | 52.3 | 52.3 |
| D-FINE-X (Hugging Face port) | 55.8 | 55.8 |

### Tutorial reproduction

| model | images | tutorial table | ngdet rerun | this repo, tutorial mode |
|---|---|---|---|---|
| torchvision:fasterrcnn_resnet50_fpn_v2 | kitti (120) | - | 0.2362 | 0.2362 |
| torchvision:fasterrcnn_resnet50_fpn_v2 | coco (120) | 0.581 | 0.5807 | 0.5807 |
| yolo:yolo11x.pt | kitti (120) | - | 0.2351 | 0.2351 |
| yolo:yolo11x.pt | coco (120) | 0.573 | 0.5727 | 0.5727 |
| gdino:IDEA-Research/grounding-dino-tiny | kitti (120) | - | 0.1861 | - |
| gdino:IDEA-Research/grounding-dino-tiny | coco (120) | 0.527 | 0.5310 | - |
| torchvision:fasterrcnn_resnet50_fpn_v2 | nuimages (500) | 0.299 | 0.2987 | 0.2987 |
| yolo:yolo11x.pt | nuimages (500) | 0.249 | 0.2486 | 0.2486 |
| gdino:IDEA-Research/grounding-dino-tiny | nuimages (500) | 0.306 | 0.3031 | - |
| torchvision:fasterrcnn_resnet50_fpn_v2 | kitti_train120 (120) | 0.279 | 0.2792 | 0.2792 |
| yolo:yolo11x.pt | kitti_train120 (120) | 0.260 | 0.2607 | 0.2607 |
| gdino:IDEA-Research/grounding-dino-tiny | kitti_train120 (120) | 0.223 | 0.2260 | - |

### Protocol audit (mean Shapley)

| dataset | mean mAP, tutorial | mean mAP, corrected | semantics | ignore | minsize | empty | threshold | fusion |
|---|---|---|---|---|---|---|---|---|
| COCO | 0.447 | 0.556 | +0.069 | +0.017 | +0.000 | +0.000 | +0.023 | +0.000 |
| KITTI | 0.213 | 0.267 | +0.004 | +0.008 | +0.006 | +0.000 | +0.007 | +0.030 |
| nuImages | 0.276 | 0.326 | +0.014 | +0.012 | +0.000 | -0.000 | +0.019 | +0.006 |

### KITTI val per class

| model | AP | AP50 | AP75 | vehicle | person | cyclist |
|---|---|---|---|---|---|---|
| Faster R-CNN | 0.256 | 0.550 | 0.218 | 0.475 | 0.225 | 0.069 |
| RetinaNet | 0.259 | 0.548 | 0.229 | 0.491 | 0.208 | 0.079 |
| FCOS | 0.245 | 0.518 | 0.217 | 0.495 | 0.140 | 0.102 |
| Faster R-CNN v2 | 0.286 | 0.601 | 0.250 | 0.504 | 0.247 | 0.108 |
| YOLO11-N | 0.224 | 0.455 | 0.201 | 0.450 | 0.191 | 0.029 |
| YOLO11-M | 0.268 | 0.559 | 0.236 | 0.499 | 0.230 | 0.074 |
| YOLO11-X | 0.272 | 0.569 | 0.240 | 0.504 | 0.232 | 0.081 |
| D-FINE-N | 0.232 | 0.459 | 0.213 | 0.489 | 0.140 | 0.068 |
| D-FINE-M | 0.276 | 0.552 | 0.245 | 0.523 | 0.188 | 0.118 |
| D-FINE-X | 0.283 | 0.564 | 0.255 | 0.528 | 0.199 | 0.123 |
| Grounding DINO-T | 0.313 | 0.666 | 0.267 | 0.496 | 0.200 | 0.244 |
| MM-GDINO-T | 0.269 | 0.606 | 0.219 | 0.476 | 0.158 | 0.175 |
| LLMDet-T | 0.287 | 0.628 | 0.239 | 0.478 | 0.217 | 0.167 |
| YOLO11-M at 1280 | 0.286 | 0.607 | 0.250 | 0.506 | 0.239 | 0.112 |
| D-FINE-M letterboxed | 0.263 | 0.546 | 0.232 | 0.506 | 0.189 | 0.095 |

### nuImages val per class

| model | AP | AP50 | AP75 | vehicle | person | cyclist |
|---|---|---|---|---|---|---|
| Faster R-CNN | 0.313 | 0.529 | 0.316 | 0.385 | 0.250 | 0.303 |
| RetinaNet | 0.297 | 0.494 | 0.293 | 0.381 | 0.209 | 0.301 |
| FCOS | 0.282 | 0.445 | 0.283 | 0.401 | 0.131 | 0.312 |
| Faster R-CNN v2 | 0.391 | 0.621 | 0.396 | 0.453 | 0.315 | 0.405 |
| YOLO11-N | 0.221 | 0.359 | 0.213 | 0.329 | 0.157 | 0.177 |
| YOLO11-M | 0.333 | 0.511 | 0.330 | 0.431 | 0.248 | 0.319 |
| YOLO11-X | 0.367 | 0.555 | 0.369 | 0.457 | 0.277 | 0.369 |
| D-FINE-N | 0.248 | 0.410 | 0.238 | 0.367 | 0.141 | 0.235 |
| D-FINE-M | 0.326 | 0.502 | 0.325 | 0.449 | 0.212 | 0.317 |
| D-FINE-X | 0.343 | 0.527 | 0.341 | 0.460 | 0.229 | 0.339 |
| Grounding DINO-T | 0.390 | 0.578 | 0.410 | 0.465 | 0.253 | 0.452 |
| MM-GDINO-T | 0.350 | 0.543 | 0.365 | 0.461 | 0.230 | 0.359 |
| LLMDet-T | 0.378 | 0.570 | 0.395 | 0.454 | 0.276 | 0.406 |
| YOLO11-M at 1280 | 0.404 | 0.608 | 0.410 | 0.478 | 0.333 | 0.403 |
| D-FINE-M letterboxed | 0.289 | 0.460 | 0.284 | 0.405 | 0.190 | 0.271 |

### Fine-tuning

| model | evaluated on | fine-tuned mAP (3 seeds) | change vs zero-shot | range of per-seed 95% CIs |
|---|---|---|---|---|
| Faster R-CNN v2 | COCO | 0.231 ± 0.008 | -0.346 | [-0.362, -0.331] |
| Faster R-CNN v2 | KITTI | 0.467 ± 0.001 | +0.181 | [+0.117, +0.216] |
| Faster R-CNN v2 | nuImages | 0.231 ± 0.003 | -0.160 | [-0.176, -0.142] |
| YOLO11-M | COCO | 0.056 ± 0.003 | -0.553 | [-0.568, -0.541] |
| YOLO11-M | KITTI | 0.373 ± 0.006 | +0.105 | [+0.056, +0.146] |
| YOLO11-M | nuImages | 0.091 ± 0.004 | -0.241 | [-0.260, -0.211] |

### Latency

| model | params (M) | 1600x900 fp32 ms | 1600x900 fp16 ms | network only fp16 ms | KITTI fp16 ms | peak memory fp16 (MB) |
|---|---|---|---|---|---|---|
| tv:fasterrcnn_resnet50_fpn | 41.76 | 30.69 | 25.58 | 13.66 | 12.43 | 637 |
| tv:retinanet_resnet50_fpn | 34.01 | 24.75 | 20.88 | 14.79 | 14.64 | 425 |
| tv:fcos_resnet50_fpn | 32.27 | 25.17 | 22.58 | 14.63 | 15.78 | 418 |
| tv:fasterrcnn_resnet50_fpn_v2 | 43.71 | 27.23 | 20.01 | 13.37 | 14.37 | 583 |
| yolo:yolo11n.pt | 2.62 | 7.33 | 7.29 | 3.98 | 6.47 | 37 |
| yolo:yolo11m.pt | 20.11 | 7.67 | 7.77 | 4.93 | 7.84 | 170 |
| yolo:yolo11x.pt | 56.97 | 9.95 | 10.42 | 7.68 | 10.85 | 561 |
| yolo:yolo11m.pt@1280 | 20.11 | 9.9 | 8.83 | 4.85 | 7.96 | 660 |
| dfine:n | 3.75 | 22.89 | 25.97 | 15.46 | 22.85 | 72 |
| dfine:m | 19.25 | 20.66 | 24.92 | 14.6 | 20.01 | 158 |
| dfine:x | 61.6 | 29.23 | 36.83 | 25.45 | 32.83 | 382 |
| dfine:m@letterbox | 19.25 | 20.45 | 23.9 | 14.65 | 19.05 | 158 |
| gdino:IDEA-Research/grounding-dino-tiny | 172.25 | 121.85 | 109.73 | 65.71 | 73.35 | 1966 |
| gdino:openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_grit_v3det | 172.91 | 119.06 | 99.33 | 66.28 | 75.2 | 1969 |
| gdino:iSEE-Laboratory/llmdet_tiny | 172.91 | 118.9 | 97.8 | 65.48 | 71.53 | 1969 |

### Paired contrasts

| A | B | dataset | AP(A) - AP(B) | 95% CI |
|---|---|---|---|---|
| yolo:yolo11m.pt@1280 | yolo:yolo11m.pt | kitti_val | +0.018 | [+0.007, +0.040] |
| yolo:yolo11m.pt@1280 | yolo:yolo11m.pt | nuimages_val | +0.072 | [+0.061, +0.084] |
| dfine:m@letterbox | dfine:m | kitti_val | -0.013 | [-0.027, -0.003] |
| dfine:m@letterbox | dfine:m | nuimages_val | -0.037 | [-0.046, -0.031] |
| gdino:iSEE-Laboratory/llmdet_tiny | gdino:openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_grit_v3det | coco_val | +0.007 | [+0.001, +0.012] |
| gdino:iSEE-Laboratory/llmdet_tiny | gdino:openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_grit_v3det | kitti_val | +0.018 | [+0.006, +0.041] |
| gdino:iSEE-Laboratory/llmdet_tiny | gdino:openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_grit_v3det | nuimages_val | +0.029 | [+0.014, +0.038] |
| gdino:IDEA-Research/grounding-dino-tiny | tv:fasterrcnn_resnet50_fpn_v2 | kitti_val | +0.027 | [-0.003, +0.044] |
| gdino:IDEA-Research/grounding-dino-tiny | tv:fasterrcnn_resnet50_fpn_v2 | nuimages_val | -0.001 | [-0.016, +0.011] |
| yolo:yolo11x.pt | dfine:x | coco_val | +0.016 | [+0.011, +0.021] |
| yolo:yolo11x.pt | dfine:x | kitti_val | -0.011 | [-0.022, +0.003] |
| yolo:yolo11x.pt | dfine:x | nuimages_val | +0.025 | [+0.013, +0.035] |
| yolo:yolo11x.pt | tv:fasterrcnn_resnet50_fpn_v2 | coco_val | +0.059 | [+0.053, +0.064] |
| yolo:yolo11x.pt | tv:fasterrcnn_resnet50_fpn_v2 | kitti_val | -0.014 | [-0.030, -0.005] |
| yolo:yolo11x.pt | tv:fasterrcnn_resnet50_fpn_v2 | nuimages_val | -0.023 | [-0.039, -0.009] |
| tv:fcos_resnet50_fpn | tv:fasterrcnn_resnet50_fpn | coco_val | +0.017 | [+0.013, +0.021] |
| tv:fcos_resnet50_fpn | tv:fasterrcnn_resnet50_fpn | kitti_val | -0.011 | [-0.046, -0.000] |
| tv:fcos_resnet50_fpn | tv:fasterrcnn_resnet50_fpn | nuimages_val | -0.031 | [-0.043, -0.020] |
| tv:retinanet_resnet50_fpn | tv:fasterrcnn_resnet50_fpn | coco_val | -0.017 | [-0.021, -0.012] |
| tv:retinanet_resnet50_fpn | tv:fasterrcnn_resnet50_fpn | kitti_val | +0.003 | [-0.011, +0.008] |
| tv:retinanet_resnet50_fpn | tv:fasterrcnn_resnet50_fpn | nuimages_val | -0.016 | [-0.025, -0.007] |
| tv:fasterrcnn_resnet50_fpn_v2 | tv:fasterrcnn_resnet50_fpn | coco_val | +0.087 | [+0.082, +0.093] |
| tv:fasterrcnn_resnet50_fpn_v2 | tv:fasterrcnn_resnet50_fpn | kitti_val | +0.030 | [+0.020, +0.049] |
| tv:fasterrcnn_resnet50_fpn_v2 | tv:fasterrcnn_resnet50_fpn | nuimages_val | +0.078 | [+0.068, +0.086] |

### Pedestrian subgroups

| model | child gap, pts (Holm p) | child gap at matched height | construction-worker gap (Holm p) | worker gap at matched height |
|---|---|---|---|---|
| Faster R-CNN | +1.6 (p=1.00) | -4.3 | +9.7 (p=0.00) | +11.6 |
| RetinaNet | +0.6 (p=1.00) | -5.3 | +7.8 (p=0.00) | +9.8 |
| FCOS | +4.1 (p=1.00) | +0.7 | +3.7 (p=0.55) | +5.0 |
| Faster R-CNN v2 | +1.8 (p=1.00) | -3.9 | +9.7 (p=0.00) | +11.5 |
| YOLO11-N | -1.5 (p=1.00) | -7.5 | +7.7 (p=0.00) | +9.5 |
| YOLO11-M | -2.3 (p=1.00) | -8.4 | +9.1 (p=0.00) | +11.4 |
| YOLO11-X | -1.3 (p=1.00) | -7.8 | +9.7 (p=0.00) | +12.0 |
| D-FINE-N | +4.7 (p=1.00) | +0.5 | +5.4 (p=0.00) | +7.0 |
| D-FINE-M | +4.7 (p=1.00) | -0.8 | +7.0 (p=0.00) | +8.9 |
| D-FINE-X | +2.1 (p=1.00) | -4.1 | +7.7 (p=0.00) | +9.7 |
| Grounding DINO-T | +4.1 (p=1.00) | -1.7 | +9.9 (p=0.00) | +11.6 |
| MM-GDINO-T | +3.0 (p=1.00) | -2.8 | +14.2 (p=0.00) | +15.7 |
| LLMDet-T | +1.7 (p=1.00) | -4.5 | +9.7 (p=0.00) | +11.5 |
| YOLO11-M fine-tuned (seed 0) | +3.0 (p=1.00) | +0.8 | +8.2 (p=0.00) | +8.5 |
| YOLO11-M fine-tuned (seed 1) | +3.8 (p=1.00) | +1.8 | +8.5 (p=0.00) | +8.7 |
| YOLO11-M fine-tuned (seed 2) | +3.2 (p=1.00) | +1.1 | +8.3 (p=0.00) | +8.6 |
| Faster R-CNN v2 fine-tuned (seed 0) | +8.3 (p=0.00) | +2.4 | +10.4 (p=0.00) | +12.6 |
| Faster R-CNN v2 fine-tuned (seed 1) | +10.7 (p=0.00) | +4.9 | +10.4 (p=0.00) | +12.6 |
| Faster R-CNN v2 fine-tuned (seed 2) | +9.3 (p=0.12) | +3.2 | +10.9 (p=0.00) | +13.1 |

