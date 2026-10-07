"""Draw the 21 landmarks on a BGR preview frame.

Colors are OpenCV BGR. Joints below `MIN_LANDMARK_CONFIDENCE` are omitted
so a missed point does not appear stuck in the corner.
"""

import cv2
import numpy as np

from airos.tracking.landmarks import (
    HAND_CONNECTIONS,
    MIN_LANDMARK_CONFIDENCE,
    Hand,
    normalized_to_pixel,
)

BONE_COLOR = (255, 196, 64)
JOINT_COLOR = (40, 220, 255)
TEXT_COLOR = (245, 245, 245)
PANEL_COLOR = (24, 24, 24)


def draw_hands(frame: np.ndarray, hands: list[Hand]) -> None:
    height, width = frame.shape[:2]
    radius = max(3, int(round(width / 320)))
    thickness = max(1, radius // 2)
    for hand in hands:
        points = {
            landmark.joint: normalized_to_pixel(landmark.x, landmark.y, width, height)
            for landmark in hand.landmarks
            if landmark.confidence >= MIN_LANDMARK_CONFIDENCE
        }
        for start, end in HAND_CONNECTIONS:
            if start in points and end in points:
                cv2.line(frame, points[start], points[end], BONE_COLOR, thickness, cv2.LINE_AA)
        for joint, (px, py) in points.items():
            cv2.circle(frame, (px, py), radius, JOINT_COLOR, -1, cv2.LINE_8)
            label = str(int(joint))
            origin = (min(width - 1, px + radius + 2), max(12, py - radius - 2))
            cv2.putText(
                frame,
                label,
                origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )


def draw_status(frame: np.ndarray, lines: list[str]) -> None:
    y = 28
    for line in lines:
        (text_width, text_height), _baseline = cv2.getTextSize(
            line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        cv2.rectangle(frame, (8, y - text_height - 8), (20 + text_width, y + 8), PANEL_COLOR, -1)
        cv2.putText(
            frame,
            line,
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )
        y += text_height + 16


def draw_banner(frame: np.ndarray, text: str) -> None:
    """Large label along the bottom, where a short wave result is easy to see."""
    height, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(1.0, width / 900)
    thickness = max(2, int(round(scale * 2)))
    (text_width, text_height), _baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(8, (width - text_width) // 2)
    y = height - max(24, text_height)
    cv2.rectangle(
        frame,
        (x - 16, y - text_height - 16),
        (x + text_width + 16, y + 16),
        PANEL_COLOR,
        -1,
    )
    cv2.putText(frame, text, (x, y), font, scale, JOINT_COLOR, thickness, cv2.LINE_AA)


def hand_status(hands: list[Hand]) -> str:
    if not hands:
        return "No hand detected"
    parts = []
    for hand in hands:
        visible = hand.count_visible()
        parts.append(
            f"{hand.handedness.label} hand | {visible}/21 landmarks | "
            f"confidence {hand.confidence:.2f}"
        )
    return "   ".join(parts)
