import pytest

from airos.tracking.landmarks import (
    HAND_CONNECTIONS,
    Hand,
    Handedness,
    Joint,
    Landmark,
    clamp01,
    hand_span,
    handedness_in_user_view,
    normalized_to_pixel,
    select_usable_hands,
    vision_to_image,
)


def test_joint_count_is_21() -> None:
    assert len(Joint) == 21
    assert [int(joint) for joint in Joint] == list(range(21))


def test_connections_reference_real_joints() -> None:
    valid = set(Joint)
    assert HAND_CONNECTIONS
    for start, end in HAND_CONNECTIONS:
        assert start in valid
        assert end in valid
        assert start != end


def test_vision_origin_is_converted_to_top_left() -> None:
    assert vision_to_image(0.0, 0.0) == (0.0, 1.0)
    assert vision_to_image(0.0, 1.0) == (0.0, 0.0)
    assert vision_to_image(1.0, 1.0) == (1.0, 0.0)
    assert vision_to_image(0.25, 0.75) == (0.25, 0.25)


def test_pixels_follow_the_actual_frame_size() -> None:
    assert normalized_to_pixel(0.0, 0.0, 320, 240) == (0, 0)
    assert normalized_to_pixel(1.0, 1.0, 320, 240) == (319, 239)
    assert normalized_to_pixel(0.5, 0.25, 640, 480) == (320, 120)
    assert normalized_to_pixel(0.5, 0.25, 100, 50) == (50, 12)


def test_pixels_reject_empty_frames() -> None:
    with pytest.raises(ValueError):
        normalized_to_pixel(0.2, 0.2, 0, 10)


def test_out_of_range_coordinates_are_clamped() -> None:
    assert clamp01(-0.2) == 0.0
    assert clamp01(1.4) == 1.0
    assert normalized_to_pixel(-1.0, 2.0, 200, 100) == (0, 99)


def test_mirrored_frame_flips_reported_handedness() -> None:
    assert handedness_in_user_view(Handedness.LEFT, True) is Handedness.RIGHT
    assert handedness_in_user_view(Handedness.RIGHT, True) is Handedness.LEFT
    assert handedness_in_user_view(Handedness.UNKNOWN, True) is Handedness.UNKNOWN
    assert handedness_in_user_view(Handedness.LEFT, False) is Handedness.LEFT


def _hand_at(*points: tuple[Joint, float, float]) -> Hand:
    placed = {joint: (x, y) for joint, x, y in points}
    landmarks = []
    for joint in Joint:
        if joint in placed:
            x, y = placed[joint]
            landmarks.append(Landmark(joint, x, y, 1.0))
        else:
            landmarks.append(Landmark(joint, 0.0, 0.0, 0.0))
    return Hand(tuple(landmarks), Handedness.RIGHT)


def test_hand_span_uses_the_larger_axis_of_visible_joints() -> None:
    hand = _hand_at((Joint.WRIST, 0.10, 0.20), (Joint.MIDDLE_TIP, 0.40, 0.25))
    assert hand_span(hand) == pytest.approx(0.30)


def test_small_detections_are_not_usable_hands() -> None:
    tiny = _hand_at((Joint.WRIST, 0.64, 0.30), (Joint.INDEX_TIP, 0.67, 0.33))
    usable = _hand_at((Joint.WRIST, 0.40, 0.40), (Joint.MIDDLE_TIP, 0.55, 0.70))
    assert select_usable_hands([tiny, usable]) == [usable]
    assert select_usable_hands([tiny], min_span=0) == [tiny]


def test_hand_requires_21_ordered_landmarks() -> None:
    landmarks = tuple(
        Landmark(joint, 0.1, 0.2, 0.8) for joint in Joint
    )
    hand = Hand(landmarks, Handedness.RIGHT)
    assert hand.landmark(Joint.INDEX_TIP).joint is Joint.INDEX_TIP
    assert hand.count_visible() == 21

    with pytest.raises(ValueError):
        Hand(landmarks[:-1], Handedness.LEFT)

    swapped = list(landmarks)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(ValueError):
        Hand(tuple(swapped), Handedness.LEFT)
