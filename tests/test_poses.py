import math

import pytest

from airos.gestures.poses import Gesture, GestureEngine, GestureReading, classify_pose
from airos.tracking.landmarks import Hand, Handedness, Joint, Landmark

_ANGLES = {
    "index": -0.35,
    "middle": 0.0,
    "ring": 0.30,
    "little": 0.58,
}
_JOINTS = {
    "index": (Joint.INDEX_MCP, Joint.INDEX_TIP),
    "middle": (Joint.MIDDLE_MCP, Joint.MIDDLE_TIP),
    "ring": (Joint.RING_MCP, Joint.RING_TIP),
    "little": (Joint.LITTLE_MCP, Joint.LITTLE_TIP),
}


def _place(angle: float, distance: float, rotation: float) -> tuple[float, float]:
    absolute = angle + rotation
    return math.sin(absolute) * distance, -math.cos(absolute) * distance


def build_hand(
    ratios: dict[str, float],
    *,
    scale: float = 0.20,
    rotation: float = 0.0,
    thumb_gap: float | None = None,
    weak: set[Joint] | None = None,
) -> Hand:
    """A schematic hand. `ratios` are tip distance / knuckle distance.

    `thumb_gap` places the thumb tip that many palm-lengths from the index tip.
    """
    points: dict[Joint, tuple[float, float]] = {Joint.WRIST: (0.0, 0.0)}
    for name, ratio in ratios.items():
        mcp, tip = _JOINTS[name]
        points[mcp] = _place(_ANGLES[name], scale, rotation)
        points[tip] = _place(_ANGLES[name], scale * ratio, rotation)
    index_tip = points[Joint.INDEX_TIP]
    if thumb_gap is None:
        points[Joint.THUMB_TIP] = _place(-1.15, scale * 1.5, rotation)
    else:
        offset = _place(0.4, scale * thumb_gap, rotation)
        points[Joint.THUMB_TIP] = (index_tip[0] + offset[0], index_tip[1] + offset[1])

    hidden = weak or set()
    landmarks = []
    for joint in Joint:
        if joint in points:
            x, y = points[joint]
            confidence = 0.05 if joint in hidden else 0.95
            landmarks.append(Landmark(joint, x, y, confidence))
        else:
            landmarks.append(Landmark(joint, 0.0, 0.0, 0.0))
    return Hand(tuple(landmarks), Handedness.RIGHT)


def _open(**kwargs) -> Hand:
    return build_hand({"index": 2.0, "middle": 2.0, "ring": 2.0, "little": 2.0}, **kwargs)


def _closed(**kwargs) -> Hand:
    return build_hand({"index": 1.1, "middle": 1.1, "ring": 1.1, "little": 1.1}, **kwargs)


def _pointing(**kwargs) -> Hand:
    return build_hand({"index": 2.0, "middle": 1.1, "ring": 1.1, "little": 1.1}, **kwargs)


def test_open_palm_closed_hand_and_pointing() -> None:
    assert classify_pose(_open()).gesture is Gesture.OPEN_PALM
    assert classify_pose(_closed()).gesture is Gesture.CLOSED_HAND
    assert classify_pose(_pointing()).gesture is Gesture.POINTING


def test_pose_does_not_depend_on_size_or_rotation() -> None:
    upright = classify_pose(_open(scale=0.12))
    turned = classify_pose(_open(scale=0.40, rotation=1.3))
    assert upright.gesture is Gesture.OPEN_PALM
    assert turned.gesture is upright.gesture
    assert upright.confidence == pytest.approx(turned.confidence)


def test_pinch_is_thumb_and_index_touching_not_a_fist() -> None:
    pinch = classify_pose(_pointing(thumb_gap=0.12))
    assert pinch.gesture is Gesture.PINCH
    assert pinch.confidence == pytest.approx((0.55 - 0.12) / 0.55)

    fist = classify_pose(_closed(thumb_gap=0.05))
    assert fist.gesture is Gesture.CLOSED_HAND


def test_ok_sign_is_a_pinch_not_an_open_palm() -> None:
    hand = build_hand(
        {"index": 2.0, "middle": 2.0, "ring": 2.0, "little": 2.0},
        thumb_gap=0.10,
    )
    assert classify_pose(hand).gesture is Gesture.PINCH


def test_half_bent_fingers_are_not_a_pose() -> None:
    hand = build_hand({"index": 1.45, "middle": 1.45, "ring": 1.45, "little": 1.45})
    assert classify_pose(hand).gesture is Gesture.NONE


def test_missing_fingertip_is_not_a_pose() -> None:
    hand = _open(weak={Joint.INDEX_TIP})
    assert classify_pose(hand).gesture is Gesture.NONE
    assert classify_pose(None).gesture is Gesture.NONE


def test_status_text_matches_the_preview() -> None:
    reading = GestureReading(Gesture.PINCH, 0.936)
    assert reading.status_lines() == ["Detected: PINCH", "Confidence: 0.94"]
    assert GestureReading(Gesture.NONE, 0.0).status_lines() == ["Detected: none"]


def test_label_waits_for_a_stable_pose() -> None:
    engine = GestureEngine(stable_frames=3)
    pointing = _pointing()
    opened = _open()

    assert engine.update(pointing).gesture is Gesture.NONE
    assert engine.update(pointing).gesture is Gesture.NONE
    assert engine.update(pointing).gesture is Gesture.POINTING

    assert engine.update(opened).gesture is Gesture.POINTING
    assert engine.update(opened).gesture is Gesture.POINTING
    assert engine.update(opened).gesture is Gesture.OPEN_PALM
