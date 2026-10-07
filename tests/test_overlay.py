import numpy as np

from airos.preview.overlay import JOINT_COLOR, draw_hands, hand_status
from airos.tracking.landmarks import Hand, Handedness, Joint, Landmark, normalized_to_pixel


def _hand_with_only_wrist_visible() -> Hand:
    landmarks = []
    for joint in Joint:
        if joint is Joint.WRIST:
            landmarks.append(Landmark(joint, 0.5, 0.5, 0.95))
        else:
            landmarks.append(Landmark(joint, 0.0, 0.0, 0.0))
    return Hand(tuple(landmarks), Handedness.RIGHT)


def test_overlay_draws_a_confident_joint_at_its_pixel() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    draw_hands(frame, [_hand_with_only_wrist_visible()])
    px, py = normalized_to_pixel(0.5, 0.5, 320, 240)
    assert frame[py, px].tolist() == list(JOINT_COLOR)


def test_overlay_hides_low_confidence_joints() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    draw_hands(frame, [_hand_with_only_wrist_visible()])
    assert frame[0, 0].tolist() == [0, 0, 0]


def test_status_line_counts_visible_landmarks() -> None:
    assert hand_status([]) == "No hand detected"
    text = hand_status([_hand_with_only_wrist_visible()])
    assert "Right hand" in text
    assert "1/21 landmarks" in text
