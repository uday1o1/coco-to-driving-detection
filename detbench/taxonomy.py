"""Label mapping from each dataset and each detector into one driving taxonomy.

Unified classes: vehicle, person, cyclist. A cyclist is a person riding a
bicycle, rider and bicycle in one box, which is how KITTI labels it and how
nuImages labels a bicycle that has a rider.

Ground truth goes through `gt_rule`, which is controlled by two switches that
the protocol audit toggles independently:

  semantics  "corrected": a class is scored only where every dataset and the
             COCO vocabulary define it the same way; other object classes are
             candidates for ignoring. "tutorial": the label folding of the
             ngdet harness (taxonomy.py, driving3), where every nuImages
             bicycle is a cyclist, motorcycles are vehicles and strollers are
             persons.
  ignore     on: candidate classes and each dataset's own ignore labels
             (KITTI DontCare, nuImages bicycle racks, COCO crowd boxes) become
             ignore regions; a detection inside one is neither a true nor a
             false positive. off: they are dropped, and COCO crowd boxes are
             scored like ordinary boxes, as in ngdet.

The result is (unified class or None, scope) with scope None (scored box, or
dropped if the class is None), "all" (ignore for every class) or "class"
(ignore for the box's own class only).
"""
from __future__ import annotations

CLASSES = ("vehicle", "person", "cyclist")
CID = {c: i for i, c in enumerate(CLASSES)}

# ngdet taxonomy.py TAXONOMY_PRESETS["driving3"], copied verbatim.
TUTORIAL_SYNONYMS = {}
for _cls, _syns in {
    "vehicle": ["vehicle", "car", "truck", "bus", "van", "trailer",
                "construction_vehicle", "construction vehicle", "tram", "train",
                "motorcycle", "motorbike"],
    "person": ["person", "pedestrian", "person_sitting", "person sitting"],
    "cyclist": ["cyclist", "bicycle", "bike"],
}.items():
    for _s in [_cls, *_syns]:
        TUTORIAL_SYNONYMS[_s] = _cls

# ------------------------------------------------------- corrected semantics

CORRECTED = {
    "kitti": {"Car": "vehicle", "Van": "vehicle", "Truck": "vehicle", "Tram": "vehicle",
              "Pedestrian": "person", "Cyclist": "cyclist"},
    "coco": {"car": "vehicle", "bus": "vehicle", "truck": "vehicle", "train": "vehicle",
             "person": "person"},
    "nuimages": {
        "vehicle.car": "vehicle", "vehicle.truck": "vehicle",
        "vehicle.bus.rigid": "vehicle", "vehicle.bus.bendy": "vehicle",
        "vehicle.emergency.ambulance": "vehicle", "vehicle.emergency.police": "vehicle",
        "human.pedestrian.adult": "person", "human.pedestrian.child": "person",
        "human.pedestrian.construction_worker": "person",
        "human.pedestrian.police_officer": "person",
        "vehicle.bicycle.with_rider": "cyclist",
    },
}
# Object classes with no consistent counterpart: ignored under corrected
# semantics (when ignore is on), dropped otherwise.
CORRECTED_IGNORE = {
    "kitti": {"Misc", "Person_sitting"},
    "coco": {"bicycle", "motorcycle"},
    "nuimages": {"vehicle.bicycle.without_rider", "vehicle.bicycle",
                 "vehicle.motorcycle", "vehicle.trailer", "vehicle.construction",
                 "human.pedestrian.personal_mobility", "human.pedestrian.stroller",
                 "human.pedestrian.wheelchair"},
}
# Ignore labels the datasets define themselves (COCO crowd is handled per box).
NATIVE_IGNORE = {"kitti": {"DontCare"}, "nuimages": {"static_object.bicycle_rack"}, "coco": set()}


def _tutorial_class(dataset: str, fine: str):
    if dataset == "nuimages":
        # ngdet datasets.nuimages_name_to_unified: dotted tokens matched right
        # to left; the rider attribute suffix is not part of the native name.
        name = fine.replace(".with_rider", "").replace(".without_rider", "")
        for tok in reversed(name.split(".")):
            if tok in TUTORIAL_SYNONYMS:
                return TUTORIAL_SYNONYMS[tok]
        return None
    return TUTORIAL_SYNONYMS.get(fine.lower())


def gt_rule(dataset: str, fine: str, crowd: bool, semantics: str, ignore: bool):
    if fine in NATIVE_IGNORE[dataset]:
        return None, ("all" if ignore else None)
    if semantics == "tutorial":
        cls = _tutorial_class(dataset, fine)
    else:
        cls = CORRECTED[dataset].get(fine)
        if cls is None and fine in CORRECTED_IGNORE[dataset]:
            return None, ("all" if ignore else None)
    if cls is not None and crowd:
        return cls, ("class" if ignore else None)
    return cls, None


# Minimum box height in pixels below which a dataset does not reliably label
# objects: KITTI's benchmark ignores boxes under 25 px, nuImages annotators
# skip objects under 10 px. Ground truth below it is ignored and detections
# below it are dropped, as the KITTI evaluation does.
MIN_HEIGHT = {"kitti": 25.0, "nuimages": 10.0, "coco": 0.0}

# ----------------------------------------------------------------- predictions

# COCO detector label -> unified class under corrected semantics. Predicted
# motorcycles have no scored counterpart and are dropped.
PRED_CORRECTED = {"car": "vehicle", "bus": "vehicle", "truck": "vehicle",
                  "train": "vehicle", "person": "person", "bicycle": "cyclist"}

# Open-vocabulary prompt terms. Each maps to the COCO-style label it stands
# for, so open-vocabulary output goes through exactly the same mapping and
# rider fusion as closed-set output.
PROMPTS = {
    "synonyms": {"car": "car", "truck": "truck", "bus": "bus", "van": "car",
                 "tram": "train", "person": "person", "pedestrian": "person",
                 "cyclist": "cyclist", "bicycle": "bicycle", "motorcycle": "motorcycle"},
    "bare": {"vehicle": "vehicle", "person": "person", "cyclist": "cyclist"},
    "rider": {"car": "car", "truck": "truck", "bus": "bus", "person": "person",
              "person riding a bicycle": "cyclist", "bicycle": "bicycle",
              "motorcycle": "motorcycle"},
}


# Some checkpoints label COCO classes with the older Pascal VOC spellings.
COCO_ALIASES = {"motorbike": "motorcycle", "aeroplane": "airplane", "sofa": "couch",
                "pottedplant": "potted plant", "diningtable": "dining table", "tvmonitor": "tv"}


def pred_class(label: str, semantics: str):
    """Unified class for a detector label (COCO name or unified name)."""
    label = label.strip().lower()
    label = COCO_ALIASES.get(label, label)
    if label in CLASSES:
        return label
    if semantics == "tutorial":
        return TUTORIAL_SYNONYMS.get(label)
    return PRED_CORRECTED.get(label)
