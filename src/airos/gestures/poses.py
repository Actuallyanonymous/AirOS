"""Static pose labels from one hand.

Distances are divided by the palm size (wrist to middle knuckle), so a hand
close to the camera and a hand farther away use the same thresholds.

A finger is extended when its tip is much farther from the wrist than its
knuckle, and folded when the tip has come back toward the wrist. That test
does not care which way the hand is rotated in the image.

These labels are not macOS actions. A pinch shown here does not click.
"""

import time
from dataclasses import dataclass

from airos.gestures.types import Gesture, GestureReading
from airos.gestures.wave import WaveDetector, WaveThresholds
from airos.tracking.landmarks import MIN_LANDMARK_CONFIDENCE, Hand, Joint, Landmark, clamp01


@dataclass(frozen=True)
class PoseThresholds:
    """Tunable cutoffs. Lengths are fractions of the palm, not pixels.

    `folded_ratio` and `extended_ratio` are tip-to-wrist distance divided by
    knuckle-to-wrist distance. The gap between them is a dead zone, so a
    half-bent finger is neither pose.
    """

    folded_ratio: float = 1.25
    extended_ratio: float = 1.70
    pinch_max_ratio: float = 0.55
    min_part_confidence: float = 0.65
    min_pinch_confidence: float = 0.45
    pinch_min_index_extension: float = 0.40
    min_joint_confidence: float = MIN_LANDMARK_CONFIDENCE

    def __post_init__(self) -> None:
        if self.extended_ratio <= self.folded_ratio:
            raise ValueError("extended_ratio must be greater than folded_ratio")
        if self.pinch_max_ratio <= 0:
            raise ValueError("pinch_max_ratio must be positive")


_FINGERS: tuple[tuple[str, Joint, Joint], ...] = (
    ("index", Joint.INDEX_MCP, Joint.INDEX_TIP),
    ("middle", Joint.MIDDLE_MCP, Joint.MIDDLE_TIP),
    ("ring", Joint.RING_MCP, Joint.RING_TIP),
    ("little", Joint.LITTLE_MCP, Joint.LITTLE_TIP),
)


def classify_pose(
    hand: Hand | None,
    thresholds: PoseThresholds | None = None,
) -> GestureReading:
    """Label the pose in this frame. `none` means no clear pose."""
    if hand is None:
        return GestureReading(Gesture.NONE, 0.0)
    limits = thresholds or PoseThresholds()
    extensions = {
        name: _finger_extension(hand, mcp, tip, limits) for name, mcp, tip in _FINGERS
    }
    pinch = _pinch(hand, extensions.get("index"), limits)
    if pinch is not None:
        return pinch
    if any(value is None for value in extensions.values()):
        return GestureReading(Gesture.NONE, 0.0)

    index = extensions["index"]
    middle = extensions["middle"]
    ring = extensions["ring"]
    little = extensions["little"]
    assert index is not None and middle is not None and ring is not None and little is not None

    extended = (index, middle, ring, little)
    if all(value >= limits.min_part_confidence for value in extended):
        return GestureReading(Gesture.OPEN_PALM, min(extended))

    others_folded = (
        1.0 - middle >= limits.min_part_confidence
        and 1.0 - ring >= limits.min_part_confidence
        and 1.0 - little >= limits.min_part_confidence
    )
    if index >= limits.min_part_confidence and others_folded:
        confidence = min(index, 1.0 - middle, 1.0 - ring, 1.0 - little)
        return GestureReading(Gesture.POINTING, confidence)

    folded = tuple(1.0 - value for value in extended)
    if all(value >= limits.min_part_confidence for value in folded):
        return GestureReading(Gesture.CLOSED_HAND, min(folded))

    return GestureReading(Gesture.NONE, 0.0)


class GestureStabilizer:
    """Keep the on-screen label from flickering between frames.

    A new pose is shown only after it wins several frames in a row.
    """

    def __init__(self, stable_frames: int = 3) -> None:
        if stable_frames < 1:
            raise ValueError("stable_frames must be at least 1")
        self._needed = stable_frames
        self._current = GestureReading(Gesture.NONE, 0.0)
        self._pending = Gesture.NONE
        self._count = 0

    def update(self, reading: GestureReading) -> GestureReading:
        if reading.gesture is self._current.gesture:
            self._pending = reading.gesture
            self._count = 0
            self._current = reading
            return self._current
        if reading.gesture is self._pending:
            self._count += 1
        else:
            self._pending = reading.gesture
            self._count = 1
        if self._count >= self._needed:
            self._current = reading
        return self._current


class GestureEngine:
    """Poses and waves. A wave is one event, then a short hold. No macOS actions."""

    def __init__(
        self,
        thresholds: PoseThresholds | None = None,
        stable_frames: int = 3,
        wave_thresholds: WaveThresholds | None = None,
    ) -> None:
        self._thresholds = thresholds or PoseThresholds()
        self._stabilizer = GestureStabilizer(stable_frames)
        self._waves = WaveDetector(wave_thresholds)
        self._held: GestureReading | None = None
        self._hold_until = 0.0

    def update(self, hand: Hand | None, timestamp: float | None = None) -> GestureReading:
        now = time.perf_counter() if timestamp is None else timestamp
        pose = self._stabilizer.update(classify_pose(hand, self._thresholds))
        event = self._waves.update(now, _wave_anchor(hand))
        if event is not None:
            self._held = event
            self._hold_until = now + self._waves.thresholds.cooldown
        if self._held is not None and now < self._hold_until:
            return self._held
        self._held = None
        return pose


def _wave_anchor(hand: Hand | None) -> tuple[float, float] | None:
    """Knuckles move with a wave. The wrist often stays nearly still.

    A right-hand wave is a swing of the palm, so the knuckle line is the
    point we follow. The wrist is only a fallback when the knuckles are hidden.
    """
    if hand is None:
        return None
    knuckles = []
    for joint in (Joint.INDEX_MCP, Joint.MIDDLE_MCP, Joint.RING_MCP):
        landmark = hand.landmark(joint)
        if landmark.confidence >= MIN_LANDMARK_CONFIDENCE:
            knuckles.append((landmark.x, landmark.y))
    if len(knuckles) >= 2:
        count = float(len(knuckles))
        return (
            sum(point[0] for point in knuckles) / count,
            sum(point[1] for point in knuckles) / count,
        )
    wrist = hand.landmark(Joint.WRIST)
    if wrist.confidence >= MIN_LANDMARK_CONFIDENCE:
        return (wrist.x, wrist.y)
    if knuckles:
        return knuckles[0]
    return None


def _pinch(
    hand: Hand,
    index_extension: float | None,
    thresholds: PoseThresholds,
) -> GestureReading | None:
    if index_extension is None or index_extension < thresholds.pinch_min_index_extension:
        return None
    thumb_tip = hand.landmark(Joint.THUMB_TIP)
    index_tip = hand.landmark(Joint.INDEX_TIP)
    if not _confident(thumb_tip, index_tip, minimum=thresholds.min_joint_confidence):
        return None
    scale = _palm_size(hand, thresholds)
    if scale is None:
        return None
    gap = _distance(thumb_tip, index_tip) / scale
    if gap >= thresholds.pinch_max_ratio:
        return None
    confidence = clamp01((thresholds.pinch_max_ratio - gap) / thresholds.pinch_max_ratio)
    if confidence < thresholds.min_pinch_confidence:
        return None
    return GestureReading(Gesture.PINCH, confidence)


def _finger_extension(
    hand: Hand,
    mcp: Joint,
    tip: Joint,
    thresholds: PoseThresholds,
) -> float | None:
    wrist = hand.landmark(Joint.WRIST)
    knuckle = hand.landmark(mcp)
    fingertip = hand.landmark(tip)
    if not _confident(wrist, knuckle, fingertip, minimum=thresholds.min_joint_confidence):
        return None
    base = _distance(wrist, knuckle)
    if base < 1e-6:
        return None
    ratio = _distance(wrist, fingertip) / base
    span = thresholds.extended_ratio - thresholds.folded_ratio
    return clamp01((ratio - thresholds.folded_ratio) / span)


def _palm_size(hand: Hand, thresholds: PoseThresholds) -> float | None:
    wrist = hand.landmark(Joint.WRIST)
    if wrist.confidence < thresholds.min_joint_confidence:
        return None
    middle = hand.landmark(Joint.MIDDLE_MCP)
    if middle.confidence >= thresholds.min_joint_confidence:
        length = _distance(wrist, middle)
        return length if length >= 1e-6 else None
    lengths: list[float] = []
    for joint in (Joint.INDEX_MCP, Joint.RING_MCP, Joint.LITTLE_MCP):
        knuckle = hand.landmark(joint)
        if knuckle.confidence < thresholds.min_joint_confidence:
            continue
        length = _distance(wrist, knuckle)
        if length >= 1e-6:
            lengths.append(length)
    if not lengths:
        return None
    lengths.sort()
    return lengths[len(lengths) // 2]


def _confident(*landmarks: Landmark, minimum: float) -> bool:
    return all(landmark.confidence >= minimum for landmark in landmarks)


def _distance(start: Landmark, end: Landmark) -> float:
    return ((start.x - end.x) ** 2 + (start.y - end.y) ** 2) ** 0.5
