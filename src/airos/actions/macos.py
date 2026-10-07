"""macOS controls used by the action engine.

Pointer and keys go through Quartz. Scrolling uses a continuous pixel
scroll, which browsers and documents actually follow. Screenshots use the
system `screencapture` tool. Play/pause is the keyboard media key, held
down long enough for macOS to notice it.
"""

import subprocess
import sys
import time
from pathlib import Path

import Quartz
from AppKit import NSEvent, NSScreen

# Virtual key codes from HIToolbox/Events.h. Quartz does not export the names.
_KEY_TAB = 0x30
_KEY_3 = 0x14
_KEY_COMMAND = 0x37
_KEY_SHIFT = 0x38
_MEDIA_PLAY = 16


class MacPerformer:
    """Send one system action. Methods return a HUD label, or None."""

    def __init__(self, margin: float = 0.18) -> None:
        self._margin = margin
        self._reported = False

    def scroll(self, lines: int) -> str:
        # Line ticks are ignored by a lot of apps. A short continuous pixel
        # gesture is what a trackpad sends, and it moves the front window.
        pixels = int(lines) * 90
        source = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
        self._scroll_phase(source, Quartz.kCGScrollPhaseMayBegin, 0)
        self._scroll_phase(source, Quartz.kCGScrollPhaseBegan, pixels)
        self._scroll_phase(source, Quartz.kCGScrollPhaseChanged, pixels)
        self._scroll_phase(source, Quartz.kCGScrollPhaseEnded, 0)
        return "Scroll up" if lines > 0 else "Scroll down"

    def next_app(self) -> str:
        self._hotkey(_KEY_TAB, command=True, shift=False)
        return "Next app"

    def previous_app(self) -> str:
        self._hotkey(_KEY_TAB, command=True, shift=True)
        return "Previous app"

    def screenshot(self) -> str:
        moment = time.strftime("%Y-%m-%d at %-I.%M.%S %p")
        path = Path.home() / "Desktop" / f"Screenshot {moment}.png"
        try:
            result = subprocess.run(
                ["/usr/sbin/screencapture", "-x", str(path)],
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
            )
        except subprocess.TimeoutExpired:
            self._explain(
                "AirOS needs Screen Recording permission to take screenshots.\n"
                "Enable it in System Settings → Privacy & Security → Screen Recording "
                "for Cursor, Terminal, or Python."
            )
            return "Screenshot"
        if result.returncode != 0 or not path.is_file() or path.stat().st_size == 0:
            self._explain_screen_recording(result)
        return "Screenshot"

    def play_pause(self) -> str:
        self._media_key(_MEDIA_PLAY)
        return "Play/pause"

    def move_cursor(self, x: float, y: float) -> None:
        px, py = self._to_screen(x, y)
        event = Quartz.CGEventCreateMouseEvent(
            None, Quartz.kCGEventMouseMoved, (px, py), Quartz.kCGMouseButtonLeft
        )
        self._post(event)

    def mouse_down(self) -> str:
        self._button(Quartz.kCGEventLeftMouseDown)
        return "Click"

    def mouse_up(self) -> str:
        self._button(Quartz.kCGEventLeftMouseUp)
        return "Click"

    def _button(self, kind) -> None:
        # Stay where the cursor already is. Quartz uses the event location,
        # so read the current global position from a fresh moved event's last point
        # by reusing the pointer location the system already has.
        event = Quartz.CGEventCreate(None)
        location = Quartz.CGEventGetLocation(event)
        button = Quartz.CGEventCreateMouseEvent(None, kind, location, Quartz.kCGMouseButtonLeft)
        self._post(button)

    def _to_screen(self, x: float, y: float) -> tuple[float, float]:
        """Map a camera point onto the main display. Camera y grows downward."""
        frame = NSScreen.mainScreen().frame()
        width = float(frame.size.width)
        height = float(frame.size.height)
        margin = self._margin
        span = max(0.05, 1.0 - 2.0 * margin)
        sx = _clamp01((x - margin) / span)
        sy = _clamp01((y - margin) / span)
        px = frame.origin.x + sx * max(0.0, width - 1.0)
        # Global event coordinates grow upward from the bottom of the display.
        py = frame.origin.y + (1.0 - sy) * max(0.0, height - 1.0)
        return (px, py)

    def _hotkey(self, key: int, *, command: bool, shift: bool) -> None:
        flags = 0
        if command:
            flags |= Quartz.kCGEventFlagMaskCommand
        if shift:
            flags |= Quartz.kCGEventFlagMaskShift
        source = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
        if shift:
            self._key(source, _KEY_SHIFT, True, flags)
        if command:
            self._key(source, _KEY_COMMAND, True, flags)
        self._key(source, key, True, flags)
        time.sleep(0.05)
        self._key(source, key, False, flags)
        if command:
            self._key(source, _KEY_COMMAND, False, 0)
        if shift:
            self._key(source, _KEY_SHIFT, False, 0)

    def _key(self, source, key: int, down: bool, flags: int) -> None:
        event = Quartz.CGEventCreateKeyboardEvent(source, key, down)
        Quartz.CGEventSetFlags(event, flags)
        self._post(event)

    def _media_key(self, key: int) -> None:
        # Down and up have to be separate events. Posting them back to back
        # is often dropped, so the key is held for a moment.
        for down in (True, False):
            data = (key << 16) | ((0xA if down else 0xB) << 8)
            event = NSEvent.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_(
                14,
                (0, 0),
                0xA00 if down else 0xB00,
                0,
                0,
                None,
                8,
                data,
                -1,
            )
            posted = event.CGEvent() if event is not None else None
            if posted is None:
                self._explain("AirOS could not send the play/pause key.")
                return
            self._post(posted)
            if down:
                time.sleep(0.06)

    def _scroll_phase(self, source, phase: int, pixels: int) -> None:
        event = Quartz.CGEventCreateScrollWheelEvent(
            source, Quartz.kCGScrollEventUnitPixel, 1, int(pixels)
        )
        if event is None:
            self._explain("AirOS could not scroll the front window.")
            return
        Quartz.CGEventSetIntegerValueField(event, Quartz.kCGScrollWheelEventScrollPhase, int(phase))
        Quartz.CGEventSetIntegerValueField(event, Quartz.kCGScrollWheelEventIsContinuous, 1)
        Quartz.CGEventSetIntegerValueField(event, Quartz.kCGScrollWheelEventPointDeltaAxis1, int(pixels))
        self._post(event)

    def _explain_screen_recording(self, result: subprocess.CompletedProcess[str]) -> None:
        detail = (result.stderr or result.stdout or "").strip()
        message = (
            "AirOS needs Screen Recording permission to take screenshots.\n"
            "Enable it in System Settings → Privacy & Security → Screen Recording "
            "for Cursor, Terminal, or Python."
        )
        if detail:
            message = f"{message}\n{detail}"
        if self._reported:
            return
        print(message, file=sys.stderr)
        self._reported = True
        subprocess.run(
            [
                "open",
                "x-apple.systempreferences:com.apple.settings.PrivacySecurity.extension?Privacy_ScreenCapture",
            ],
            check=False,
        )

    def _explain(self, message: str) -> None:
        if self._reported:
            return
        print(message, file=sys.stderr)
        self._reported = True

    def _post(self, event) -> None:
        try:
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
        except Exception as exc:
            if not self._reported:
                print(f"AirOS could not control the Mac: {exc}", file=sys.stderr)
                self._reported = True


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value
