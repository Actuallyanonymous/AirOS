"""Hand landmarks in image coordinates.

The tracker always produces 21 points, in `Joint` order. Gesture code should
use the enum, not raw indexes.

Coordinates are normalized to the camera frame:

- origin is the top-left
- x increases to the right
- y increases downward
- both axes are in the range 0–1

The frame width and height are applied only when drawing or hitting a pixel.
Nothing here assumes a webcam resolution or a hand size.
"""

from dataclasses import dataclass
from enum import IntEnum


class Joint(IntEnum):
    """21 hand joints, wrist first, then each finger from base to tip."""

    WRIST = 0
    THUMB_CMC = 1
    THUMB_MP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    LITTLE_MCP = 17
    LITTLE_PIP = 18
    LITTLE_DIP = 19
    LITTLE_TIP = 20


# Bones drawn between joints. Palm links connect the finger bases.
HAND_CONNECTIONS: tuple[tuple[Joint, Joint], ...] = (
    (Joint.WRIST, Joint.THUMB_CMC),
    (Joint.THUMB_CMC, Joint.THUMB_MP),
    (Joint.THUMB_MP, Joint.THUMB_IP),
    (Joint.THUMB_IP, Joint.THUMB_TIP),
    (Joint.WRIST, Joint.INDEX_MCP),
    (Joint.INDEX_MCP, Joint.INDEX_PIP),
    (Joint.INDEX_PIP, Joint.INDEX_DIP),
    (Joint.INDEX_DIP, Joint.INDEX_TIP),
    (Joint.WRIST, Joint.MIDDLE_MCP),
    (Joint.MIDDLE_MCP, Joint.MIDDLE_PIP),
    (Joint.MIDDLE_PIP, Joint.MIDDLE_DIP),
    (Joint.MIDDLE_DIP, Joint.MIDDLE_TIP),
    (Joint.WRIST, Joint.RING_MCP),
    (Joint.RING_MCP, Joint.RING_PIP),
    (Joint.RING_PIP, Joint.RING_DIP),
    (Joint.RING_DIP, Joint.RING_TIP),
    (Joint.WRIST, Joint.LITTLE_MCP),
    (Joint.LITTLE_MCP, Joint.LITTLE_PIP),
    (Joint.LITTLE_PIP, Joint.LITTLE_DIP),
    (Joint.LITTLE_DIP, Joint.LITTLE_TIP),
    (Joint.INDEX_MCP, Joint.MIDDLE_MCP),
    (Joint.MIDDLE_MCP, Joint.RING_MCP),
    (Joint.RING_MCP, Joint.LITTLE_MCP),
)

# Apple's hand-pose sample ignores joints below this score.
MIN_LANDMARK_CONFIDENCE = 0.3

# A hand held toward the webcam is much larger than this. Smaller detections
# are usually a face, poster, or other texture, not a hand you can gesture with.
MIN_HAND_SPAN = 0.08


class Handedness(IntEnum):
    """Which of the user's hands this is, after mirror correction."""

    UNKNOWN = 0
    LEFT = 1
    RIGHT = 2

    @property
    def label(self) -> str:
        if self is Handedness.LEFT:
            return "Left"
        if self is Handedness.RIGHT:
            return "Right"
        return "Unknown"


@dataclass(frozen=True)
class Landmark:
    joint: Joint
    x: float
    y: float
    confidence: float


@dataclass(frozen=True)
class Hand:
    landmarks: tuple[Landmark, ...]
    handedness: Handedness

    def __post_init__(self) -> None:
        expected = len(Joint)
        if len(self.landmarks) != expected:
            raise ValueError(f"Expected {expected} landmarks, got {len(self.landmarks)}")
        for index, landmark in enumerate(self.landmarks):
            if int(landmark.joint) != index:
                raise ValueError("Landmarks must be ordered wrist, thumb, index, middle, ring, little")

    def landmark(self, joint: Joint) -> Landmark:
        return self.landmarks[int(joint)]

    @property
    def confidence(self) -> float:
        """Mean confidence across all 21 joints, including weak ones."""
        return sum(landmark.confidence for landmark in self.landmarks) / len(self.landmarks)

    def count_visible(self, threshold: float = MIN_LANDMARK_CONFIDENCE) -> int:
        return sum(1 for landmark in self.landmarks if landmark.confidence >= threshold)


def vision_to_image(x: float, y: float) -> tuple[float, float]:
    """Convert a Vision point to top-left image coordinates.

    Vision's origin is the bottom-left and y increases upward.
    """
    return x, 1.0 - y


def clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def normalized_to_pixel(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    """Map normalized image coordinates onto a frame of any size."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    x = clamp01(x)
    y = clamp01(y)
    px = min(width - 1, int(x * width))
    py = min(height - 1, int(y * height))
    return px, py


def hand_span(hand: Hand, threshold: float = MIN_LANDMARK_CONFIDENCE) -> float:
    """Largest width or height of the visible landmarks, as a fraction of the frame."""
    visible = [landmark for landmark in hand.landmarks if landmark.confidence >= threshold]
    if len(visible) < 2:
        return 0.0
    xs = [landmark.x for landmark in visible]
    ys = [landmark.y for landmark in visible]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def select_usable_hands(hands: list[Hand], min_span: float = MIN_HAND_SPAN) -> list[Hand]:
    """Drop detections too small to be a hand in front of the camera.

    `min_span` of 0 keeps every detection. That is useful when tuning the preview.
    """
    if min_span <= 0:
        return list(hands)
    return [hand for hand in hands if hand_span(hand) >= min_span]


def handedness_in_user_view(reported: Handedness, frame_is_mirrored: bool) -> Handedness:
    """Undo the chirality flip introduced by mirroring the camera frame.

    A mirrored image makes the user's right hand look like a left hand to the
    pose model. The preview is mirrored so movement matches a mirror, and the
    label should still name the user's actual hand.
    """
    if not frame_is_mirrored or reported is Handedness.UNKNOWN:
        return reported
    if reported is Handedness.LEFT:
        return Handedness.RIGHT
    return Handedness.LEFT
