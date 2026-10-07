"""AirOS hand-tracking preview.

Opens the webcam, names a pose or a wave, and performs the mapped Mac action.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

from airos import __version__
from airos.actions.engine import ActionEngine
from airos.actions.macos import MacPerformer
from airos.config.settings import load_settings
from airos.gestures.poses import GestureEngine
from airos.camera.camera import (
    PREFERRED_HEIGHT,
    PREFERRED_WIDTH,
    Camera,
    CameraError,
    camera_by_index,
    default_camera_index,
    list_cameras,
)
from airos.permissions.macos import (
    ACCESSIBILITY_MESSAGE,
    CameraPermissionError,
    accessibility_trusted,
    ensure_camera_permission,
    open_accessibility_settings,
)
from airos.gestures.types import Gesture
from airos.preview.overlay import draw_banner, draw_hands, draw_status, hand_status
from airos.tracking.hand_tracker import HandTracker, HandTrackerError
from airos.tracking.landmarks import MIN_HAND_SPAN

WINDOW_NAME = "AirOS"
_MISSED_FRAMES_LIMIT = 30
_WAVES = {
    Gesture.WAVE_UP,
    Gesture.WAVE_DOWN,
    Gesture.WAVE_LEFT,
    Gesture.WAVE_RIGHT,
}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.list_cameras:
            _print_cameras()
            return 0
        return _run_preview(args)
    except (CameraError, CameraPermissionError, HandTrackerError) as exc:
        print(exc, file=sys.stderr)
        return 1


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="airos",
        description="Watch the webcam and perform the mapped Mac actions.",
    )
    parser.add_argument("--list-cameras", action="store_true", help="Print cameras and exit")
    parser.add_argument("--camera", type=int, default=None, help="Camera index from --list-cameras")
    parser.add_argument("--width", type=int, default=PREFERRED_WIDTH, help="Requested frame width")
    parser.add_argument("--height", type=int, default=PREFERRED_HEIGHT, help="Requested frame height")
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Show the raw camera instead of a mirror image",
    )
    parser.add_argument("--no-window", action="store_true", help="Run without opening a preview window")
    parser.add_argument(
        "--no-actions",
        action="store_true",
        help="Show gestures without controlling the Mac",
    )
    parser.add_argument(
        "--min-span",
        type=float,
        default=MIN_HAND_SPAN,
        help="Ignore hands smaller than this fraction of the frame (0 keeps every detection)",
    )
    parser.add_argument("--frames", type=int, default=None, help="Stop after this many frames")
    parser.add_argument("--save", type=Path, default=None, help="Save the last annotated frame to this path")
    args = parser.parse_args(argv)
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive")
    if args.frames is not None and args.frames <= 0:
        parser.error("--frames must be positive")
    if args.min_span < 0:
        parser.error("--min-span must be zero or positive")
    return args


def _print_cameras() -> None:
    cameras = list_cameras()
    if not cameras:
        print("AirOS could not find a webcam.")
        return
    for camera in cameras:
        print(f"{camera.index}  {camera.name}  ({camera.kind})")


def _run_preview(args: argparse.Namespace) -> int:
    ensure_camera_permission()
    cameras = list_cameras()
    index = default_camera_index(cameras) if args.camera is None else args.camera
    info = camera_by_index(cameras, index)
    mirror = not args.no_mirror

    print(f"AirOS {__version__}")
    print(f"Camera: {info.name}")
    print("Loading the on-device hand model…")
    tracker = HandTracker(max_hands=1, min_span=args.min_span)
    gestures = GestureEngine()
    settings = load_settings()
    actions, action_status = _start_actions(args, settings)
    tracker.warmup()

    frames = 0
    frames_with_hand = 0
    detect_ms = 0.0
    fps = 0.0
    previous = time.perf_counter()
    missed = 0
    last_frame = None
    window_open = False
    action_label: str | None = None

    try:
        with Camera(info, width=args.width, height=args.height) as camera:
            if not args.no_window:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                window_open = True

            try:
                while args.frames is None or frames < args.frames:
                    raw = camera.read()
                    if raw is None:
                        missed += 1
                        if missed >= _MISSED_FRAMES_LIMIT:
                            raise CameraError("AirOS lost the camera feed.")
                        continue
                    missed = 0

                    frame = cv2.flip(raw, 1) if mirror else raw.copy()
                    now = time.perf_counter()
                    hands = tracker.detect(frame, frame_is_mirrored=mirror)
                    reading = gestures.update(hands[0] if hands else None, now)
                    if actions is not None:
                        action_label = actions.update(reading, hands[0] if hands else None, now)
                    detect_ms += (time.perf_counter() - now) * 1000.0

                    elapsed = now - previous
                    previous = now
                    if elapsed > 0:
                        instantaneous = 1.0 / elapsed
                        fps = instantaneous if fps == 0 else (0.9 * fps + 0.1 * instantaneous)

                    height, width = frame.shape[:2]
                    view = "Mirrored preview" if mirror else "Raw camera"
                    draw_hands(frame, hands)
                    if reading.gesture in _WAVES:
                        draw_banner(frame, reading.gesture.label)
                    status = [
                        f"AirOS  {info.name}  {width}x{height}  {fps:.0f} fps",
                        *reading.status_lines(),
                        hand_status(hands),
                    ]
                    if action_label:
                        status.append(f"Action: {action_label}")
                    elif action_status:
                        status.append(action_status)
                    status.append(f"{view} | q or esc to quit")
                    draw_status(frame, status)
                    last_frame = frame
                    frames += 1
                    if hands:
                        frames_with_hand += 1

                    if window_open:
                        cv2.imshow(WINDOW_NAME, frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), 27) or _window_closed():
                            break
            except KeyboardInterrupt:
                pass

            if args.save is not None and last_frame is not None:
                if not cv2.imwrite(str(args.save), last_frame):
                    raise CameraError(f"AirOS could not save the preview frame to {args.save}.")
                print(f"Saved {args.save}")
    finally:
        if window_open:
            cv2.destroyAllWindows()

    average = detect_ms / frames if frames else 0.0
    print(f"Frames: {frames}")
    print(f"Frames with a hand: {frames_with_hand}")
    print(f"Detection: {average:.1f} ms average")
    return 0


def _start_actions(args, settings):
    if args.no_actions:
        print("Actions are off.")
        return None, "Actions off"
    if not accessibility_trusted():
        print(ACCESSIBILITY_MESSAGE, file=sys.stderr)
        open_accessibility_settings()
        return None, "Needs Accessibility permission"
    print("Actions on. Accessibility permission is granted.")
    return ActionEngine(MacPerformer(margin=settings.camera_margin), settings), None


def _window_closed() -> bool:
    try:
        return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True
