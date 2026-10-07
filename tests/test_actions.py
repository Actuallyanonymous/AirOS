from airos.actions.engine import ActionEngine, ActionSettings
from airos.gestures.types import Gesture, GestureReading
from airos.tracking.landmarks import Hand, Handedness, Joint, Landmark


class Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def scroll(self, lines: int) -> None:
        self.calls.append(("scroll", lines))

    def next_app(self) -> None:
        self.calls.append(("next_app",))

    def previous_app(self) -> None:
        self.calls.append(("previous_app",))

    def screenshot(self) -> None:
        self.calls.append(("screenshot",))

    def play_pause(self) -> None:
        self.calls.append(("play_pause",))

    def move_cursor(self, x: float, y: float) -> None:
        self.calls.append(("move", x, y))

    def mouse_down(self) -> None:
        self.calls.append(("down",))

    def mouse_up(self) -> None:
        self.calls.append(("up",))


def _reading(gesture: Gesture) -> GestureReading:
    return GestureReading(gesture, 1.0)


def _tip(x: float, y: float) -> Hand:
    landmarks = []
    for joint in Joint:
        if joint is Joint.INDEX_TIP:
            landmarks.append(Landmark(joint, x, y, 0.95))
        else:
            landmarks.append(Landmark(joint, 0.0, 0.0, 0.0))
    return Hand(tuple(landmarks), Handedness.RIGHT)


def _engine(recorder: Recorder, **kwargs) -> ActionEngine:
    return ActionEngine(recorder, ActionSettings(**kwargs))


def test_one_wave_scrolls_once_until_the_next_wave() -> None:
    recorder = Recorder()
    engine = _engine(recorder)
    wave = _reading(Gesture.WAVE_UP)
    engine.update(wave, None, 0.0)
    engine.update(wave, None, 0.2)
    engine.update(wave, None, 0.4)
    assert recorder.calls == [("scroll", 4)]
    engine.update(_reading(Gesture.NONE), None, 1.0)
    engine.update(wave, None, 1.2)
    assert recorder.calls == [("scroll", 4), ("scroll", 4)]


def test_wave_directions_map_to_scroll_and_apps() -> None:
    recorder = Recorder()
    engine = _engine(recorder)
    engine.update(_reading(Gesture.WAVE_DOWN), None, 0.0)
    engine.update(_reading(Gesture.NONE), None, 0.2)
    engine.update(_reading(Gesture.WAVE_RIGHT), None, 0.4)
    engine.update(_reading(Gesture.NONE), None, 0.6)
    engine.update(_reading(Gesture.WAVE_LEFT), None, 0.8)
    assert recorder.calls == [("scroll", -4), ("next_app",), ("previous_app",)]


def test_open_then_close_screenshots_but_a_fist_does_not() -> None:
    recorder = Recorder()
    engine = _engine(recorder)
    engine.update(_reading(Gesture.NONE), None, 0.0, raw=_reading(Gesture.CLOSED_HAND))
    assert recorder.calls == []
    engine.update(_reading(Gesture.NONE), None, 0.1, raw=_reading(Gesture.OPEN_PALM))
    engine.update(_reading(Gesture.NONE), None, 0.4, raw=_reading(Gesture.CLOSED_HAND))
    assert recorder.calls == [("screenshot",)]


def test_a_brief_gap_does_not_cancel_an_open_palm() -> None:
    recorder = Recorder()
    engine = _engine(recorder, palm_hold=0.6, palm_grace=0.35, close_window=1.3)
    engine.update(_reading(Gesture.NONE), None, 0.0, raw=_reading(Gesture.OPEN_PALM))
    engine.update(_reading(Gesture.NONE), None, 0.2, raw=_reading(Gesture.NONE))
    engine.update(_reading(Gesture.NONE), None, 0.35, raw=_reading(Gesture.CLOSED_HAND))
    assert recorder.calls == [("screenshot",)]

    recorder.calls.clear()
    engine = _engine(recorder, palm_hold=0.5, palm_grace=0.35)
    engine.update(_reading(Gesture.NONE), None, 0.0, raw=_reading(Gesture.OPEN_PALM))
    engine.update(_reading(Gesture.NONE), None, 0.15, raw=_reading(Gesture.NONE))
    engine.update(_reading(Gesture.NONE), None, 0.55, raw=_reading(Gesture.OPEN_PALM))
    assert recorder.calls == [("play_pause",)]


def test_holding_an_open_palm_plays_once() -> None:
    recorder = Recorder()
    engine = _engine(recorder, palm_hold=0.7, play_cooldown=1.2)
    for t in (0.0, 0.4, 0.7, 1.2, 2.0):
        engine.update(_reading(Gesture.NONE), None, t, raw=_reading(Gesture.OPEN_PALM))
    assert recorder.calls == [("play_pause",)]
    engine.update(_reading(Gesture.NONE), None, 2.2, raw=_reading(Gesture.NONE))
    engine.update(_reading(Gesture.NONE), None, 2.3, raw=_reading(Gesture.OPEN_PALM))
    engine.update(_reading(Gesture.NONE), None, 3.1, raw=_reading(Gesture.OPEN_PALM))
    assert recorder.calls == [("play_pause",), ("play_pause",)]


def test_a_long_hold_does_not_also_take_a_screenshot() -> None:
    recorder = Recorder()
    engine = _engine(recorder, palm_hold=0.5)
    engine.update(_reading(Gesture.NONE), None, 0.0, raw=_reading(Gesture.OPEN_PALM))
    engine.update(_reading(Gesture.NONE), None, 0.5, raw=_reading(Gesture.OPEN_PALM))
    engine.update(_reading(Gesture.NONE), None, 0.8, raw=_reading(Gesture.CLOSED_HAND))
    assert recorder.calls == [("play_pause",)]


def test_pinch_clicks_once_and_a_pinch_with_movement_drags() -> None:
    recorder = Recorder()
    engine = _engine(recorder)
    hand = _tip(0.40, 0.40)
    engine.update(_reading(Gesture.NONE), hand, 0.0, raw=_reading(Gesture.PINCH))
    engine.update(_reading(Gesture.NONE), hand, 0.05, raw=_reading(Gesture.PINCH))
    assert ("down",) in recorder.calls
    assert recorder.calls.count(("down",)) == 1
    engine.update(_reading(Gesture.NONE), hand, 0.1, raw=_reading(Gesture.PINCH))
    assert recorder.calls.count(("down",)) == 1
    engine.update(_reading(Gesture.NONE), None, 0.2, raw=_reading(Gesture.NONE))
    engine.update(_reading(Gesture.NONE), None, 0.25, raw=_reading(Gesture.NONE))
    assert recorder.calls.count(("up",)) == 1

    moved = _tip(0.50, 0.40)
    engine.update(_reading(Gesture.NONE), moved, 0.4, raw=_reading(Gesture.PINCH))
    engine.update(_reading(Gesture.NONE), moved, 0.45, raw=_reading(Gesture.PINCH))
    engine.update(_reading(Gesture.NONE), _tip(0.60, 0.40), 0.5, raw=_reading(Gesture.PINCH))
    assert any(call[0] == "move" and abs(call[1] - 0.40) > 0.05 for call in recorder.calls)


def test_pointing_ignores_tiny_jitter() -> None:
    recorder = Recorder()
    engine = _engine(recorder, cursor_deadzone=0.02, cursor_smoothing=1)
    engine.update(_reading(Gesture.NONE), _tip(0.40, 0.40), 0.0, raw=_reading(Gesture.POINTING))
    engine.update(_reading(Gesture.NONE), _tip(0.405, 0.40), 0.05, raw=_reading(Gesture.POINTING))
    moves = [call for call in recorder.calls if call[0] == "move"]
    assert len(moves) == 1
    engine.update(_reading(Gesture.NONE), _tip(0.50, 0.40), 0.1, raw=_reading(Gesture.POINTING))
    moves = [call for call in recorder.calls if call[0] == "move"]
    assert len(moves) == 2
