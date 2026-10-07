"""Decide when a gesture becomes a single Mac action.

Recognition stays in the gesture engine. This module only decides when to
act, so a held pose cannot fire over and over.

Open palm, then a close, is a screenshot. A fist with no open palm is not.
Holding the open palm is play/pause, once, until the hand leaves and comes back.
A pinch presses the mouse button; releasing it clicks. Moving while pinched drags.
"""

from dataclasses import dataclass

from airos.gestures.poses import classify_pose
from airos.gestures.types import Gesture, GestureReading
from airos.tracking.landmarks import MIN_LANDMARK_CONFIDENCE, Hand, Joint

_WAVES = {
    Gesture.WAVE_UP,
    Gesture.WAVE_DOWN,
    Gesture.WAVE_LEFT,
    Gesture.WAVE_RIGHT,
}

DEFAULT_MAPPINGS = {
    "wave_up": "scroll_up",
    "wave_down": "scroll_down",
    "wave_left": "previous_app",
    "wave_right": "next_app",
    "open_to_close": "screenshot",
    "open_palm_hold": "play_pause",
    "point": "move_cursor",
    "pinch": "left_click",
}

_LABELS = {
    "scroll_up": "Scroll up",
    "scroll_down": "Scroll down",
    "next_app": "Next app",
    "previous_app": "Previous app",
    "screenshot": "Screenshot",
    "play_pause": "Play/pause",
    "left_click": "Click",
    "drag": "Drag",
}


@dataclass
class ActionSettings:
    """Numbers the action layer can tune without touching gesture detection."""

    palm_hold: float = 0.55
    play_cooldown: float = 1.2
    close_window: float = 1.3
    # A single missed frame must not wipe an open palm that is about to close or finish a hold.
    palm_grace: float = 0.35
    scroll_lines: int = 4
    cursor_smoothing: float = 0.45
    cursor_deadzone: float = 0.008
    drag_slop: float = 0.025
    camera_margin: float = 0.18
    mappings: dict[str, str] | None = None

    def action_for(self, key: str) -> str | None:
        table = self.mappings if self.mappings is not None else DEFAULT_MAPPINGS
        name = table.get(key)
        if not name or name == "none":
            return None
        return name


class ActionEngine:
    """Turn gesture readings into performer calls. One gesture, one action."""

    def __init__(self, performer, settings: ActionSettings | None = None) -> None:
        self._performer = performer
        self._settings = settings or ActionSettings()
        self._active_wave: Gesture | None = None
        self._open_since: float | None = None
        self._away_since: float | None = None
        self._play_fired = False
        self._need_release = False
        self._pinch_frames = 0
        self._release_frames = 0
        self._button_down = False
        self._pinch_origin: tuple[float, float] | None = None
        self._smooth: tuple[float, float] | None = None
        self._last_label: str | None = None
        self._shown_at = 0.0
        self._play_ready = -1.0

    def update(
        self,
        reading: GestureReading,
        hand: Hand | None,
        now: float,
        raw: GestureReading | None = None,
    ) -> str | None:
        """Perform at most the action for this frame. Return a short HUD label."""
        if reading.gesture in _WAVES:
            self._wave(reading.gesture, now)
            self._cancel_pose()
        else:
            self._active_wave = None
            pose = raw.gesture if raw is not None else _pose(hand)
            self._pose(pose, hand, now)
        if self._last_label is not None and now - self._shown_at < 1.0:
            return self._last_label
        return None

    def _wave(self, gesture: Gesture, now: float) -> None:
        if gesture is self._active_wave:
            return
        self._active_wave = gesture
        name = self._settings.action_for(gesture.value)
        if name == "scroll_up":
            self._performer.scroll(self._settings.scroll_lines)
            self._note("Scroll up", now)
        elif name == "scroll_down":
            self._performer.scroll(-self._settings.scroll_lines)
            self._note("Scroll down", now)
        elif name == "next_app":
            self._performer.next_app()
            self._note("Next app", now)
        elif name == "previous_app":
            self._performer.previous_app()
            self._note("Previous app", now)

    def _pose(self, pose: Gesture, hand: Hand | None, now: float) -> None:
        if self._need_release:
            if pose is Gesture.OPEN_PALM:
                return
            self._need_release = False
            self._play_fired = False
        self._palm(pose, now)
        self._pinch(pose, hand, now)
        if pose is Gesture.POINTING and not self._button_down:
            point = _index_tip(hand)
            if point is not None and self._settings.action_for("point") == "move_cursor":
                self._move(point[0], point[1])

    def _palm(self, pose: Gesture, now: float) -> None:
        settings = self._settings
        if pose is Gesture.OPEN_PALM:
            self._away_since = None
            if self._open_since is None:
                self._open_since = now
                return
            held = now - self._open_since
            if (
                not self._play_fired
                and held >= settings.palm_hold
                and now >= self._play_ready
                and settings.action_for("open_palm_hold") == "play_pause"
            ):
                self._performer.play_pause()
                self._note("Play/pause", now)
                self._play_fired = True
                self._play_ready = now + settings.play_cooldown
                self._open_since = None
                self._need_release = True
            return

        if (
            pose is Gesture.CLOSED_HAND
            and self._saw_open(now)
            and not self._play_fired
            and settings.action_for("open_to_close") == "screenshot"
        ):
            self._performer.screenshot()
            self._note("Screenshot", now)
            self._open_since = None
            self._away_since = None
            self._need_release = True
            return

        if self._open_since is None:
            return
        if self._away_since is None:
            self._away_since = now
        elif now - self._away_since > settings.palm_grace:
            self._open_since = None
            self._away_since = None
            self._play_fired = False

    def _saw_open(self, now: float) -> bool:
        if self._open_since is None:
            return False
        return now - self._open_since <= self._settings.close_window

    def _pinch(self, pose: Gesture, hand: Hand | None, now: float) -> None:
        if self._settings.action_for("pinch") != "left_click":
            if self._button_down:
                self._performer.mouse_up()
                self._button_down = False
            return
        if pose is Gesture.PINCH:
            self._pinch_frames += 1
            self._release_frames = 0
        else:
            self._release_frames += 1
            self._pinch_frames = 0

        point = _index_tip(hand)
        if not self._button_down and self._pinch_frames >= 2:
            if point is not None:
                self._move(point[0], point[1], force=True)
                self._pinch_origin = point
            self._performer.mouse_down()
            self._note("Click", now)
            self._button_down = True
            return
        if self._button_down and point is not None and self._pinch_origin is not None:
            if _distance(point, self._pinch_origin) >= self._settings.drag_slop:
                self._move(point[0], point[1])
                self._note("Drag", now)
        if self._button_down and self._release_frames >= 2:
            self._performer.mouse_up()
            self._button_down = False
            self._pinch_origin = None

    def _move(self, x: float, y: float, force: bool = False) -> None:
        if self._smooth is not None and not force:
            if _distance((x, y), self._smooth) < self._settings.cursor_deadzone:
                return
        if self._smooth is None or force:
            self._smooth = (x, y)
        else:
            alpha = self._settings.cursor_smoothing
            sx, sy = self._smooth
            self._smooth = (alpha * x + (1.0 - alpha) * sx, alpha * y + (1.0 - alpha) * sy)
        self._performer.move_cursor(self._smooth[0], self._smooth[1])

    def _cancel_pose(self) -> None:
        self._open_since = None
        self._away_since = None
        self._play_fired = False
        self._need_release = True
        if self._button_down:
            self._performer.mouse_up()
            self._button_down = False
            self._pinch_origin = None

    def _note(self, label: str, now: float) -> None:
        self._last_label = label
        self._shown_at = now


def _pose(hand: Hand | None) -> Gesture:
    if hand is None:
        return Gesture.NONE
    return classify_pose(hand).gesture


def _index_tip(hand: Hand | None) -> tuple[float, float] | None:
    if hand is None:
        return None
    tip = hand.landmark(Joint.INDEX_TIP)
    if tip.confidence < MIN_LANDMARK_CONFIDENCE:
        return None
    return (tip.x, tip.y)


def _distance(start: tuple[float, float], end: tuple[float, float]) -> float:
    return ((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2) ** 0.5
