"""Rider fusion on hand-built scenes."""
import numpy as np

from detbench.protocol import fuse_riders


def test_rider_on_bicycle_becomes_one_cyclist():
    boxes = np.array([[100, 50, 140, 150],     # person, lower half over the bike
                      [90, 100, 160, 170],     # bicycle
                      [300, 50, 340, 150],     # pedestrian elsewhere
                      [500, 120, 570, 170]],   # parked bicycle
                     np.float32)
    scores = np.array([0.9, 0.8, 0.7, 0.6], np.float32)
    labels = np.array(["person", "bicycle", "person", "bicycle"])
    b, s, l = fuse_riders(boxes, scores, labels)
    assert sorted(l.tolist()) == ["cyclist", "person"]
    cyc = b[l == "cyclist"][0]
    assert np.allclose(cyc, [90, 50, 160, 170])
    assert np.isclose(s[l == "cyclist"][0], np.sqrt(0.9 * 0.8))


def test_rider_on_motorcycle_is_absorbed():
    boxes = np.array([[100, 50, 140, 150], [80, 100, 170, 175]], np.float32)
    b, s, l = fuse_riders(boxes, np.array([0.9, 0.9], np.float32), np.array(["person", "motorcycle"]))
    assert len(l) == 0


def test_person_beside_bicycle_is_not_paired():
    boxes = np.array([[200, 50, 240, 150], [90, 100, 160, 170]], np.float32)
    b, s, l = fuse_riders(boxes, np.array([0.9, 0.9], np.float32), np.array(["person", "bicycle"]))
    assert l.tolist() == ["person"]
