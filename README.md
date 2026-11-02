# COCO to driving: how far do 2D detectors transfer, and how much does the protocol decide?

Thirteen COCO-trained detectors from four families (two-stage CNN, one-stage CNN, real-time DETR,
open vocabulary) evaluated zero-shot on COCO val2017, KITTI val and the full nuImages validation set,
plus fine-tuning on KITTI, a latency benchmark and a pedestrian subgroup analysis. Every number comes
with a 95% confidence interval from a bootstrap that resamples whole drives (KITTI) or logs
(nuImages), and every comparison claimed in the text is a paired test on the same images.

The work starts from the zero-shot harness in the
[DeepDataMiningLearning ngdet tutorial](https://github.com/lkk688/DeepDataMiningLearning/tree/main/DeepDataMiningLearning/ngdet),
reproduces its published table exactly, and then measures how the evaluation choices it already lists
as caveats (fixed score threshold, cyclist mapping, sample size) change the answer.

## Findings

1. **The tutorial's numbers reproduce exactly.** Its own code at a pinned commit gives the published
   values on COCO, nuImages and KITTI (Faster R-CNN v2: 0.581, 0.299, 0.279). The KITTI column comes from
   the first 120 images of the KITTI *train* list. This repo's harness, run in tutorial mode on the same
   images, matches ngdet to four decimals on all eight closed-set pairs.
2. **The protocol moves the numbers as much as the model does.** Going from the tutorial protocol to a
   corrected one (ignore regions, each dataset's own minimum-size rule, no score cut-off before mAP,
   consistent class definitions, rider fusion) raises mean mAP from 0.213 to 0.267 on KITTI and from 0.276
   to 0.326 on nuImages. On KITTI the model ranking changes substantially (Kendall tau 0.36 between the
   two protocols). The fixed 0.3 score cut-off alone costs YOLO models 0.04 to 0.07 mAP on nuImages
   and COCO.
3. **COCO rankings do not carry over to driving data.** YOLO11-X leads on COCO (0.636) but loses to
   Faster R-CNN v2 on KITTI (-0.014) and nuImages (-0.023), both significant. With backbone, input size and
   training recipe held fixed, FCOS beats Faster R-CNN on COCO (+0.017) and loses to it on nuImages (-0.031).
4. **Input resolution is a hidden confound.** YOLO11-M at 1280 px instead of its default 640 gains
   +0.072 mAP on nuImages (95% CI +0.061 to +0.084) and becomes the best model there (0.404). D-FINE's
   stretched 640x640 input beats letterboxing on wide driving frames (-0.037 when letterboxed).
5. **Pedestrians are the class that breaks.** Vehicle AP on KITTI is close to COCO; person AP falls
   from about 0.6 to about 0.2 on both driving sets and stays low even for pedestrians taller than 160 px.
6. **Construction workers are missed 4 to 14 recall points more often than other adult pedestrians, by
   every model,** and the gap grows (5 to 16 points) when recall is compared at matched height, so object
   size does not explain it. Children show no significant gap zero-shot; after fine-tuning Faster R-CNN v2
   on KITTI a child gap of 8 to 11 points appears.
7. **Fine-tuning on one driving dataset hurts the other.** Fine-tuning on KITTI train (3 seeds, seed
   s.d. at most 0.008) raises KITTI val mAP by +0.18 (Faster R-CNN v2) and +0.10 (YOLO11-M) but lowers
   nuImages by -0.16 and -0.24 and COCO by -0.35 and -0.55.
8. **Open-vocabulary detectors are competitive on driving data and sensitive to the prompt.** Grounding
   DINO-T is best on KITTI (0.313). LLMDet beats its own starting point MM-GDINO on all three datasets
   (+0.007, +0.018, +0.029). Prompt wording changes mAP by up to 0.15.

![protocol audit](figures/protocol_audit.png)

| | |
|---|---|
| ![per class](figures/cross_dataset_per_class.png) | ![speed](figures/speed_accuracy.png) |
| ![fine-tuning](figures/finetune.png) | ![subgroups](figures/pedestrian_subgroups.png) |

## Main results (corrected protocol)

mAP@[.5:.95] with 95% clustered bootstrap intervals (1000 resamples). Classes: vehicle, person, cyclist
(cyclist is not scored on COCO). All tables, including per-class AP, AP50/AP75, AR, AP by object height,
the protocol factorial, latency and the subgroup tests, are in [results/tables.md](results/tables.md) and
the JSON files in [results/](results).

| model | family | COCO | KITTI | nuImages |
|---|---|---|---|---|
| Faster R-CNN (R50-FPN) | two-stage CNN | 0.490 [0.480, 0.500] | 0.256 [0.236, 0.314] | 0.313 [0.291, 0.331] |
| RetinaNet (R50-FPN) | one-stage CNN | 0.473 [0.463, 0.484] | 0.259 [0.240, 0.308] | 0.297 [0.275, 0.317] |
| FCOS (R50-FPN) | one-stage CNN | 0.507 [0.496, 0.518] | 0.245 [0.227, 0.281] | 0.282 [0.258, 0.299] |
| Faster R-CNN v2 (R50-FPN) | two-stage CNN | 0.577 [0.569, 0.587] | 0.286 [0.261, 0.364] | 0.391 [0.366, 0.409] |
| YOLO11-N | YOLO | 0.494 [0.482, 0.506] | 0.224 [0.210, 0.259] | 0.221 [0.202, 0.236] |
| YOLO11-M | YOLO | 0.609 [0.599, 0.620] | 0.268 [0.249, 0.326] | 0.333 [0.306, 0.352] |
| YOLO11-X | YOLO | 0.636 [0.625, 0.647] | 0.272 [0.253, 0.334] | 0.367 [0.339, 0.391] |
| D-FINE-N | real-time DETR | 0.498 [0.486, 0.509] | 0.232 [0.218, 0.269] | 0.248 [0.225, 0.267] |
| D-FINE-M | real-time DETR | 0.598 [0.587, 0.609] | 0.276 [0.253, 0.329] | 0.326 [0.302, 0.345] |
| D-FINE-X | real-time DETR | 0.620 [0.609, 0.631] | 0.283 [0.260, 0.342] | 0.343 [0.318, 0.361] |
| Grounding DINO-T | open vocabulary | 0.570 [0.559, 0.582] | 0.313 [0.287, 0.370] | 0.390 [0.359, 0.410] |
| MM-GDINO-T | open vocabulary | 0.578 [0.567, 0.590] | 0.269 [0.250, 0.322] | 0.350 [0.333, 0.372] |
| LLMDet-T | open vocabulary | 0.585 [0.574, 0.596] | 0.287 [0.264, 0.350] | 0.378 [0.358, 0.396] |
| YOLO11-M at 1280 px | YOLO | | 0.286 [0.261, 0.358] | 0.404 [0.380, 0.426] |
| D-FINE-M letterboxed | real-time DETR | | 0.263 [0.246, 0.310] | 0.289 [0.265, 0.308] |

Open-vocabulary COCO numbers are not strictly zero-shot: LLMDet's training data contains COCO images,
and the GoldG grounding data used by all three contains Visual Genome images, many of which are COCO
images.

## What is evaluated

**Data.** COCO val2017 (5,000 images); KITTI object val, the drive-disjoint split of Chen et al.
(NeurIPS 2015; 3,769 images from 45 drives, none shared with the 96 training drives); nuImages
v1.0-val, all 16,445 key frames from 82 logs including the 1,561 with no objects.

**Models.** Official checkpoints with their own preprocessing:

| model | source | input as deployed | post-processing |
|---|---|---|---|
| Faster R-CNN, RetinaNet, FCOS (v1 recipe), Faster R-CNN v2 | torchvision COCO weights | short side 800, long side at most 1333 | class-wise NMS 0.5 |
| YOLO11-N/M/X | Ultralytics | letterbox to 640 (aspect kept) | NMS 0.7, multi-label as in Ultralytics validation |
| D-FINE-N/M/X | authors' code (Peterande/D-FINE @ 956d170) and released COCO weights | stretch to 640x640 | top 300 (query, class) pairs |
| Grounding DINO-T, MM-GDINO-T, LLMDet-T | Hugging Face transformers | short side 800, long side at most 1333 | top 300 (query, term) pairs, term score = mean token probability |

All detections are kept down to score 0.001 (at most 300 per image; 100 per class at evaluation), so
any protocol can be applied afterwards without rerunning a model. Before any cross-dataset number is
trusted, each checkpoint must reproduce its published COCO val AP:

| | FRCNN | RetinaNet | FCOS | FRCNN v2 | YOLO11 N/M/X | D-FINE N/M/X | GDINO-T | MM-GDINO-T |
|---|---|---|---|---|---|---|---|---|
| measured | 37.2 | 35.9 | 39.1 | 47.3 | 39.3 / 51.3 / 54.5 | 42.8 / 52.3 / 55.8 | 48.5 | 50.4 |
| published | 37.0 | 36.4 | 39.2 | 46.7 | 39.5 / 51.5 / 54.7 | 42.8 / 52.3 / 55.8 | 48.4 | 50.4 |

## Evaluation protocol

Unified classes: **vehicle**, **person** and **cyclist**, where a cyclist is a person riding a bicycle
with one box around both, as KITTI and nuImages label it. A class is scored only where every dataset and
the COCO vocabulary define it the same way. Everything else becomes an ignore region: a detection inside
one is neither a true nor a false positive.

| dataset | vehicle | person | cyclist | ignored for every class |
|---|---|---|---|---|
| KITTI | Car, Van, Truck, Tram | Pedestrian | Cyclist | DontCare, Misc, Person_sitting; boxes under 25 px tall (KITTI benchmark rule) |
| nuImages | car, truck, bus, emergency | adult, child, construction worker, police officer | bicycle with rider | parked bicycles, motorcycles, bicycle racks, trailers, construction vehicles, strollers, wheelchairs, personal mobility; detections under 10 px (annotation rule) |
| COCO | car, bus, truck, train | person | not scored | bicycle, motorcycle; crowd boxes ignore their own class |

COCO-trained models see a cyclist as a person plus a bicycle. **Rider fusion** pairs a person with a
bicycle when the bicycle scores at least 0.1, the lower half of the person box overlaps the bicycle box by
at least 70% and the person's centre lies over the bicycle; the pair becomes one cyclist box. The two
thresholds were chosen on KITTI *train* predictions of one model (Faster R-CNN v2) and fixed for every
model and dataset. On nuImages, whose labels say whether each bicycle has a rider, fusion confidently pairs
49 to 92% of detected bicycles with riders and 2 to 34% of parked ones. On nuImages it raises cyclist AP
from 0.03 to 0.09 to between 0.18 and 0.45 but lowers person AP for some models, because pedestrians standing next to parked
bicycles are absorbed (results/fusion.json).

The evaluator reproduces pycocotools' matching exactly (tests/test_evaluate.py) and keeps the outcome of
every detection, so a bootstrap resample is one weighted pass instead of a full re-evaluation. Every
COCO-style table is computed twice, once with faster-coco-eval and once with this matcher, and the two
must agree.

## Reproducing

Hardware used: one RTX 4090 (24 GB), 32 CPU cores, 108 GB RAM, rented on vast.ai. GPU time for
everything was about 13 hours, most of it the open-vocabulary models on nuImages. Software: Python 3.12,
torch 2.11.0+cu128, torchvision 0.26.0, transformers 5.22.1, ultralytics 8.4.197 (requirements.txt).

```bash
git clone https://github.com/uday1o1/coco-to-driving-detection && cd coco-to-driving-detection
export DATA=$HOME/data RUNS=$HOME/runs
bash scripts/setup.sh              # dependencies, COCO val, KITTI object, nuImages val (public, no account)
bash scripts/run_zero_shot.sh      # every zero-shot run, ablation, prompt variant and COCO check
bash scripts/run_finetune.sh       # YOLO11-M and Faster R-CNN v2 on KITTI train, seeds 0-2, then evaluation
bash scripts/reproduce_tutorial.sh # ngdet at its pinned commit on the tutorial's own images
bash scripts/run_latency.sh        # on an otherwise idle GPU
bash scripts/run_analysis.sh       # CPU only: every file in results/ and figures/
```

**Without a GPU.** The stored detections and fine-tuned checkpoints are attached to the
[v1.0 release](https://github.com/uday1o1/coco-to-driving-detection/releases/tag/v1.0)
(detections.tar, 1.6 GB; finetuned_checkpoints.tar, 0.6 GB; SHA-256 sums in the release notes).
Extract detections.tar into `$RUNS`, run `NO_IMAGES=1 bash scripts/setup.sh` (annotations and metadata
only, about 1 GB) and then `scripts/run_analysis.sh`. Images are needed only for the qualitative figure.

Tests: `python -m pytest tests -q` (evaluator against pycocotools, rider fusion).

Run logs, nvidia-smi output, per-run metadata (command, versions, GPU, wall time) and training curves are
in [logs/](logs).

## Layout

```
detbench/   taxonomy.py   label mapping and ignore rules, one place
            data.py       dataset readers and protocol-specific COCO ground truth
            detectors.py  wrappers: torchvision, Ultralytics, official D-FINE, Grounding DINO family
            infer.py      run one model over one dataset, store every detection
            protocol.py   score thresholds, size rules, rider fusion applied to stored detections
            evaluate.py   exact COCO matcher, weighted AP, clustered bootstrap
            analysis.py   per-run metrics, paired differences, operating thresholds
            finetune.py   KITTI fine-tuning for YOLO11-M and Faster R-CNN v2
            latency.py    batch-1 latency, memory and parameter count
scripts/    setup and run scripts, analyze.py (every result), figures.py, tables.py, qualitative.py
results/    JSON results and tables.md
figures/    figures used here and in the report
logs/       run logs and metadata
tests/      evaluator and fusion tests
```

## Scope and limits

KITTI val has 45 drives, so its intervals are wide (about +-0.05); differences between models on KITTI
need the paired tests in results/contrasts.json. nuImages val has 192 night frames, all from one log, so
lighting is not analysed. Police officers (96) and children (251) are small groups; their intervals
reflect that. Latency is measured with each library's PyTorch path at batch 1 without TensorRT, so it
compares implementations as shipped, not optimised engines. The fine-tuning recipes are each library's
standard one (YOLO: 100 epochs with mosaic; Faster R-CNN: 12 epochs), so the two fine-tuned models
differ in recipe as well as architecture.

## Licences and credits

COCO (CC BY 4.0 annotations), KITTI (CC BY-NC-SA 3.0) and nuImages (CC BY-NC-SA 4.0) are used under their
terms; the non-commercial licences apply to anything derived from KITTI and nuImages here. Models:
torchvision, Ultralytics YOLO11, D-FINE (Peng et al., ICLR 2025), Grounding DINO (Liu et al., ECCV 2024),
MM-Grounding-DINO (Zhao et al., 2024), LLMDet (Fu et al., CVPR 2025). The tutorial harness is ngdet from
DeepDataMiningLearning.
