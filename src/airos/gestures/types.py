"""Gesture names shared by pose and wave detection."""

from dataclasses import dataclass
from enum import Enum


class Gesture(Enum):
    NONE = "none"
    OPEN_PALM = "open_palm"
    CLOSED_HAND = "closed_hand"
    PINCH = "pinch"
    POINTING = "pointing"
    WAVE_UP = "wave_up"
    WAVE_DOWN = "wave_down"
    WAVE_LEFT = "wave_left"
    WAVE_RIGHT = "wave_right"

    @property
    def label(self) -> str:
        if self is Gesture.OPEN_PALM:
            return "OPEN PALM"
        if self is Gesture.CLOSED_HAND:
            return "CLOSED HAND"
        if self is Gesture.PINCH:
            return "PINCH"
        if self is Gesture.POINTING:
            return "POINTING"
        if self is Gesture.WAVE_UP:
            return "WAVE UP"
        if self is Gesture.WAVE_DOWN:
            return "WAVE DOWN"
        if self is Gesture.WAVE_LEFT:
            return "WAVE LEFT"
        if self is Gesture.WAVE_RIGHT:
            return "WAVE RIGHT"
        return "none"


@dataclass(frozen=True)
class GestureReading:
    gesture: Gesture
    confidence: float

    def status_lines(self) -> list[str]:
        if self.gesture is Gesture.NONE:
            return ["Detected: none"]
        return [
            f"Detected: {self.gesture.label}",
            f"Confidence: {self.confidence:.2f}",
        ]
