import math

from airos.gestures.poses import Gesture, GestureEngine
from airos.gestures.wave import WaveDetector, WaveThresholds
from airos.tracking.landmarks import Hand, Handedness, Joint, Landmark


def _move(detector: WaveDetector, points: list[tuple[float, float, float]]):
    found = []
    for t, x, y in points:
        event = detector.update(t, (x, y))
        if event is not None:
            found.append(event)
    return found


def _line(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    seconds: float,
    fps: float = 30,
    t0: float = 0.0,
) -> list[tuple[float, float, float]]:
    steps = max(1, int(round(seconds * fps)))
    points = []
    for step in range(steps + 1):
        blend = step / steps
        points.append((t0 + step / fps, x0 + (x1 - x0) * blend, y0 + (y1 - y0) * blend))
    return points


def _dwell(
    x: float,
    y: float,
    seconds: float,
    t0: float,
    fps: float = 30,
) -> list[tuple[float, float, float]]:
    steps = max(1, int(round(seconds * fps)))
    return [(t0 + step / fps, x, y) for step in range(1, steps + 1)]


def _finished(points: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    """Hold still at the end so the stroke can close."""
    t, x, y = points[-1]
    return points + _dwell(x, y, 0.36, t)


def test_cardinal_waves() -> None:
    limits = WaveThresholds(smoothing=1)
    cases = (
        (_line(0.20, 0.50, 0.60, 0.50, 0.30), Gesture.WAVE_RIGHT),
        (_line(0.70, 0.50, 0.30, 0.50, 0.30), Gesture.WAVE_LEFT),
        (_line(0.50, 0.70, 0.50, 0.30, 0.30), Gesture.WAVE_UP),
        (_line(0.50, 0.30, 0.50, 0.70, 0.30), Gesture.WAVE_DOWN),
    )
    for points, expected in cases:
        found = _move(WaveDetector(limits), _finished(points))
        assert [event.gesture for event in found] == [expected]
        assert found[0].confidence > 0.6


def test_continuing_the_same_way_is_one_wave() -> None:
    limits = WaveThresholds(smoothing=1)
    first = _line(0.20, 0.50, 0.65, 0.50, 0.30)
    more = _line(0.65, 0.50, 0.90, 0.50, 0.25, t0=first[-1][0] + 1 / 30)
    found = _move(WaveDetector(limits), _finished(first + more[1:]))
    assert [event.gesture for event in found] == [Gesture.WAVE_RIGHT]


def test_a_later_swipe_is_a_second_wave() -> None:
    limits = WaveThresholds(smoothing=1)
    detector = WaveDetector(limits)
    first = _finished(_line(0.20, 0.50, 0.55, 0.50, 0.30))
    assert [event.gesture for event in _move(detector, first)] == [Gesture.WAVE_RIGHT]
    second = _finished(_line(0.55, 0.50, 0.85, 0.50, 0.30, t0=first[-1][0] + 0.05))
    assert [event.gesture for event in _move(detector, second)] == [Gesture.WAVE_RIGHT]


def test_windup_then_the_real_swipe_is_one_wave() -> None:
    """Pulling down before waving up is the preparation, not WAVE DOWN."""
    limits = WaveThresholds(smoothing=1)
    down = _line(0.50, 0.40, 0.50, 0.64, 0.24)
    up = _line(0.50, 0.64, 0.50, 0.28, 0.36, t0=down[-1][0] + 1 / 30)
    found = _move(WaveDetector(limits), _finished(down + up[1:]))
    assert [event.gesture for event in found] == [Gesture.WAVE_UP]

    left = _line(0.58, 0.50, 0.42, 0.50, 0.22)
    right = _line(0.42, 0.50, 0.74, 0.50, 0.36, t0=left[-1][0] + 1 / 30)
    found = _move(WaveDetector(limits), _finished(left + right[1:]))
    assert [event.gesture for event in found] == [Gesture.WAVE_RIGHT]


def test_a_pause_keeps_two_deliberate_waves() -> None:
    limits = WaveThresholds(smoothing=1)
    detector = WaveDetector(limits)
    left = _finished(_line(0.72, 0.50, 0.42, 0.50, 0.34))
    assert [event.gesture for event in _move(detector, left)] == [Gesture.WAVE_LEFT]
    right = _finished(_line(0.42, 0.50, 0.74, 0.50, 0.34, t0=left[-1][0] + 0.05))
    assert [event.gesture for event in _move(detector, right)] == [Gesture.WAVE_RIGHT]


def test_a_lift_before_a_sideways_swipe_stays_sideways() -> None:
    """A right hand rises, then moves right. That is WAVE RIGHT, not WAVE UP."""
    limits = WaveThresholds(smoothing=1)
    lift = _line(0.40, 0.62, 0.43, 0.46, 0.22)
    swipe = _line(0.43, 0.46, 0.70, 0.44, 0.36, t0=lift[-1][0] + 1 / 30)
    found = _move(WaveDetector(limits), _finished(lift + swipe[1:]))
    assert [event.gesture for event in found] == [Gesture.WAVE_RIGHT]


def test_a_sideways_arc_is_not_an_up_wave() -> None:
    limits = WaveThresholds(smoothing=1)
    steps = 16
    points = []
    for step in range(steps + 1):
        blend = step / steps
        x = 0.30 + 0.30 * blend
        y = 0.56 - 0.13 * math.sin(math.pi * blend)
        points.append((step / 30, x, y))
    found = _move(WaveDetector(limits), _finished(points))
    assert [event.gesture for event in found] == [Gesture.WAVE_RIGHT]


def test_small_slow_and_diagonal_motion_are_ignored() -> None:
    limits = WaveThresholds(smoothing=1)
    detector = WaveDetector(limits)
    jitter = [(step / 30, 0.50 + 0.012 * ((step % 5) - 2), 0.50) for step in range(40)]
    assert _move(detector, _finished(jitter)) == []
    assert _move(WaveDetector(limits), _finished(_line(0.40, 0.50, 0.45, 0.50, 0.20))) == []
    assert _move(WaveDetector(limits), _finished(_line(0.40, 0.50, 0.52, 0.50, 2.00))) == []
    assert _move(WaveDetector(limits), _finished(_line(0.30, 0.30, 0.60, 0.60, 0.30))) == []


def test_shaky_hand_still_ends_the_wave() -> None:
    """A held hand jitters. That must not keep the wave from finishing."""
    limits = WaveThresholds(smoothing=0.65)
    stroke = _line(0.32, 0.55, 0.58, 0.52, 0.30)
    t, x, y = stroke[-1]
    points = list(stroke)
    for step in range(1, 16):
        wobble = 0.012 if step % 2 else -0.012
        points.append((t + step / 30, x + wobble, y - wobble * 0.4))
    found = _move(WaveDetector(limits), points)
    assert [event.gesture for event in found] == [Gesture.WAVE_RIGHT]


def test_a_normal_speed_swipe_counts() -> None:
    found = _move(
        WaveDetector(WaveThresholds(smoothing=1)),
        _finished(_line(0.40, 0.50, 0.55, 0.50, 0.50)),
    )
    assert [event.gesture for event in found] == [Gesture.WAVE_RIGHT]


def test_a_brief_dropout_does_not_cancel_a_wave() -> None:
    detector = WaveDetector(WaveThresholds(smoothing=1))
    points = _finished(_line(0.20, 0.50, 0.55, 0.50, 0.40))
    found = []
    for index, (t, x, y) in enumerate(points):
        point = None if index == 6 else (x, y)
        event = detector.update(t, point)
        if event is not None:
            found.append(event)
    assert [event.gesture for event in found] == [Gesture.WAVE_RIGHT]


def test_a_gap_in_tracking_does_not_become_a_wave() -> None:
    detector = WaveDetector(WaveThresholds(smoothing=1))
    detector.update(0.0, (0.20, 0.50))
    detector.update(0.05, None)
    event = detector.update(0.08, (0.70, 0.50))
    assert event is None


def test_knuckles_count_when_the_wrist_stays_still() -> None:
    engine = GestureEngine(stable_frames=1, wave_thresholds=WaveThresholds(smoothing=1))
    saw_wave = False
    steps = 15
    last_t = 0.0
    last_x = 0.30
    for step in range(steps + 1):
        last_t = step / 30
        last_x = 0.30 + 0.25 * (step / steps)
        reading = engine.update(_hand_with_fixed_wrist(last_x), last_t)
        saw_wave = saw_wave or reading.gesture is Gesture.WAVE_RIGHT
    for extra in range(1, 12):
        reading = engine.update(_hand_with_fixed_wrist(last_x), last_t + extra / 30)
        saw_wave = saw_wave or reading.gesture is Gesture.WAVE_RIGHT
    assert saw_wave


def test_preview_holds_the_wave_then_returns_to_the_pose() -> None:
    limits = WaveThresholds(smoothing=1, cooldown=0.40)
    engine = GestureEngine(stable_frames=1, wave_thresholds=limits)
    points = _finished(_line(0.20, 0.50, 0.60, 0.50, 0.30))
    saw_wave = False
    for t, x, y in points:
        reading = engine.update(_hand_at(x, y), t)
        saw_wave = saw_wave or reading.gesture is Gesture.WAVE_RIGHT
    assert saw_wave
    end_t, end_x, end_y = points[-1]
    assert engine.update(_hand_at(end_x, end_y), end_t + 0.05).gesture is Gesture.WAVE_RIGHT
    assert engine.update(_hand_at(end_x, end_y), end_t + 0.70).gesture is Gesture.NONE


def _hand_with_fixed_wrist(knuckle_x: float) -> Hand:
    landmarks = []
    for joint in Joint:
        if joint is Joint.WRIST:
            landmarks.append(Landmark(joint, 0.40, 0.70, 0.95))
        elif joint in (Joint.INDEX_MCP, Joint.MIDDLE_MCP, Joint.RING_MCP):
            landmarks.append(Landmark(joint, knuckle_x, 0.50, 0.95))
        else:
            landmarks.append(Landmark(joint, knuckle_x, 0.50, 0.0))
    return Hand(tuple(landmarks), Handedness.RIGHT)


def _hand_at(x: float, y: float) -> Hand:
    landmarks = []
    for joint in Joint:
        confidence = 0.95 if joint is Joint.WRIST else 0.0
        landmarks.append(Landmark(joint, x, y, confidence))
    return Hand(tuple(landmarks), Handedness.RIGHT)
