"""On-device hand landmarks via Apple's Vision framework.

Vision's `VNDetectHumanHandPoseRequest` returns the same 21-joint hand used
by the rest of AirOS. The model is part of macOS, runs on the Neural Engine,
and does not send frames or usage metrics anywhere.

The request is created once and reused. The first call loads the model and
can take a few seconds; later frames are a few milliseconds on Apple Silicon.
"""

import sys

import cv2
import numpy as np
import Vision
from Foundation import NSData
from Quartz import (
    CGColorSpaceCreateDeviceRGB,
    CGDataProviderCreateWithCFData,
    CGImageCreate,
    kCGBitmapByteOrderDefault,
    kCGImageAlphaNone,
    kCGRenderingIntentDefault,
)

from airos.tracking.landmarks import (
    MIN_HAND_SPAN,
    Hand,
    Handedness,
    Joint,
    Landmark,
    clamp01,
    handedness_in_user_view,
    select_usable_hands,
    vision_to_image,
)

_BITMAP_INFO = kCGImageAlphaNone | kCGBitmapByteOrderDefault

# Joint order must match `Joint`. Vision names are resolved once at import.
_VISION_JOINTS: tuple[tuple[Joint, str], ...] = (
    (Joint.WRIST, Vision.VNHumanHandPoseObservationJointNameWrist),
    (Joint.THUMB_CMC, Vision.VNHumanHandPoseObservationJointNameThumbCMC),
    (Joint.THUMB_MP, Vision.VNHumanHandPoseObservationJointNameThumbMP),
    (Joint.THUMB_IP, Vision.VNHumanHandPoseObservationJointNameThumbIP),
    (Joint.THUMB_TIP, Vision.VNHumanHandPoseObservationJointNameThumbTip),
    (Joint.INDEX_MCP, Vision.VNHumanHandPoseObservationJointNameIndexMCP),
    (Joint.INDEX_PIP, Vision.VNHumanHandPoseObservationJointNameIndexPIP),
    (Joint.INDEX_DIP, Vision.VNHumanHandPoseObservationJointNameIndexDIP),
    (Joint.INDEX_TIP, Vision.VNHumanHandPoseObservationJointNameIndexTip),
    (Joint.MIDDLE_MCP, Vision.VNHumanHandPoseObservationJointNameMiddleMCP),
    (Joint.MIDDLE_PIP, Vision.VNHumanHandPoseObservationJointNameMiddlePIP),
    (Joint.MIDDLE_DIP, Vision.VNHumanHandPoseObservationJointNameMiddleDIP),
    (Joint.MIDDLE_TIP, Vision.VNHumanHandPoseObservationJointNameMiddleTip),
    (Joint.RING_MCP, Vision.VNHumanHandPoseObservationJointNameRingMCP),
    (Joint.RING_PIP, Vision.VNHumanHandPoseObservationJointNameRingPIP),
    (Joint.RING_DIP, Vision.VNHumanHandPoseObservationJointNameRingDIP),
    (Joint.RING_TIP, Vision.VNHumanHandPoseObservationJointNameRingTip),
    (Joint.LITTLE_MCP, Vision.VNHumanHandPoseObservationJointNameLittleMCP),
    (Joint.LITTLE_PIP, Vision.VNHumanHandPoseObservationJointNameLittlePIP),
    (Joint.LITTLE_DIP, Vision.VNHumanHandPoseObservationJointNameLittleDIP),
    (Joint.LITTLE_TIP, Vision.VNHumanHandPoseObservationJointNameLittleTip),
)

_CHIRALITY = {
    int(Vision.VNChiralityLeft): Handedness.LEFT,
    int(Vision.VNChiralityRight): Handedness.RIGHT,
}


class HandTrackerError(Exception):
    """Raised when the on-device hand model cannot be loaded."""


class HandTracker:
    """Detect up to `max_hands` hands in a BGR camera frame."""

    def __init__(self, max_hands: int = 1, min_span: float = MIN_HAND_SPAN) -> None:
        if max_hands < 1:
            raise ValueError("max_hands must be at least 1")
        if min_span < 0:
            raise ValueError("min_span must be zero or positive")
        self._min_span = min_span
        self._request = Vision.VNDetectHumanHandPoseRequest.alloc().init()
        self._request.setMaximumHandCount_(max_hands)
        self._color_space = CGColorSpaceCreateDeviceRGB()
        self._pixel_data = None
        self._provider = None
        self._failure: str | None = None
        self._reported_failure = False

    def warmup(self) -> None:
        """Load the Vision model before the preview starts.

        A blank frame is enough. No hand is expected, and no camera frame is kept.
        """
        blank = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.detect(blank, frame_is_mirrored=False)
        if self._failure is not None:
            raise HandTrackerError(
                "AirOS could not start the on-device hand tracker. " + self._failure
            )

    def detect(self, frame_bgr: np.ndarray, *, frame_is_mirrored: bool) -> list[Hand]:
        """Return hands found in this frame. An empty list means no hand.

        A single bad frame is reported once and does not stop the preview.
        """
        self._failure = None
        try:
            image = self._cgimage(frame_bgr)
            handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(image, None)
            ok, error = handler.performRequests_error_([self._request], None)
            if not ok:
                self._note_failure(error)
                return []
            observations = self._request.results()
            if observations is None or len(observations) == 0:
                return []
            hands = [
                self._hand_from_observation(observation, frame_is_mirrored)
                for observation in observations
            ]
            return select_usable_hands(hands, self._min_span)
        except Exception as exc:
            self._note_failure(exc)
            return []

    def _hand_from_observation(self, observation, frame_is_mirrored: bool) -> Hand:
        landmarks = []
        for joint, vision_name in _VISION_JOINTS:
            x, y, confidence = _read_joint(observation, vision_name)
            landmarks.append(Landmark(joint, x, y, confidence))
        reported = _CHIRALITY.get(int(observation.chirality()), Handedness.UNKNOWN)
        handedness = handedness_in_user_view(reported, frame_is_mirrored)
        return Hand(tuple(landmarks), handedness)

    def _cgimage(self, frame_bgr: np.ndarray):
        rgb = np.ascontiguousarray(_as_rgb(frame_bgr))
        height, width, _channels = rgb.shape
        pixel_data = NSData.dataWithBytes_length_(rgb.tobytes(), rgb.nbytes)
        provider = CGDataProviderCreateWithCFData(pixel_data)
        image = CGImageCreate(
            width,
            height,
            8,
            24,
            width * 3,
            self._color_space,
            _BITMAP_INFO,
            provider,
            None,
            False,
            kCGRenderingIntentDefault,
        )
        if image is None:
            raise RuntimeError("Could not convert the camera frame for hand tracking.")
        # The provider borrows these bytes. Keep them alive across performRequests,
        # which returns before the next frame replaces them.
        self._pixel_data = pixel_data
        self._provider = provider
        return image

    def _note_failure(self, error: object) -> None:
        self._failure = str(error) if error is not None else "unknown Vision error"
        if not self._reported_failure:
            print(f"AirOS: hand tracking failed on a frame: {self._failure}", file=sys.stderr)
            self._reported_failure = True


def _as_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    if frame_bgr.ndim == 2:
        return cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2RGB)
    channels = frame_bgr.shape[2]
    if channels == 4:
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGRA2RGB)
    if channels == 3:
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    raise ValueError(f"Unsupported frame shape {frame_bgr.shape}")


def _read_joint(observation, vision_name: str) -> tuple[float, float, float]:
    point, _error = observation.recognizedPointForJointName_error_(vision_name, None)
    if point is None:
        return 0.0, 0.0, 0.0
    location = point.location()
    x, y = vision_to_image(float(location.x), float(location.y))
    return clamp01(x), clamp01(y), float(point.confidence())
