"""Camera and Accessibility permission checks."""

import ctypes
import subprocess
import threading

from AVFoundation import (
    AVAuthorizationStatusAuthorized,
    AVAuthorizationStatusDenied,
    AVAuthorizationStatusNotDetermined,
    AVAuthorizationStatusRestricted,
    AVCaptureDevice,
    AVMediaTypeVideo,
)

CAMERA_PERMISSION_MESSAGE = (
    "AirOS needs Camera permission to see your hand.\n"
    "Enable Camera access in System Settings → Privacy & Security → Camera "
    "for the app that launched AirOS (Terminal, Cursor, or Python)."
)


class CameraPermissionError(Exception):
    """Raised when the webcam cannot be used because permission is missing."""


def camera_authorization_status() -> int:
    return int(AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeVideo))


def ensure_camera_permission(timeout: float = 120.0) -> None:
    """Return if the camera is authorized. Otherwise raise a clear error.

    A not-yet-decided status prompts macOS once and waits for the user.
    """
    status = camera_authorization_status()
    if status == int(AVAuthorizationStatusAuthorized):
        return
    if status == int(AVAuthorizationStatusNotDetermined):
        granted = _request_camera_access(timeout)
        if granted:
            return
        raise CameraPermissionError(CAMERA_PERMISSION_MESSAGE)
    if status in (
        int(AVAuthorizationStatusDenied),
        int(AVAuthorizationStatusRestricted),
    ):
        raise CameraPermissionError(CAMERA_PERMISSION_MESSAGE)
    raise CameraPermissionError(CAMERA_PERMISSION_MESSAGE)


def _request_camera_access(timeout: float) -> bool:
    done = threading.Event()
    result = {"granted": False}

    def handler(granted: bool) -> None:
        result["granted"] = bool(granted)
        done.set()

    AVCaptureDevice.requestAccessForMediaType_completionHandler_(AVMediaTypeVideo, handler)
    if not done.wait(timeout):
        return False
    return bool(result["granted"])


ACCESSIBILITY_MESSAGE = (
    "AirOS needs Accessibility permission to control your Mac.\n"
    "Enable it in System Settings → Privacy & Security → Accessibility "
    "for the app that launched AirOS (Terminal, Cursor, or Python)."
)

_ACCESSIBILITY_URL = (
    "x-apple.systempreferences:com.apple.settings.PrivacySecurity.extension?Privacy_Accessibility"
)


def accessibility_trusted() -> bool:
    """True when this process may post pointer and keyboard events."""
    try:
        library = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices"
        )
    except OSError:
        return False
    library.AXIsProcessTrusted.restype = ctypes.c_bool
    return bool(library.AXIsProcessTrusted())


def open_accessibility_settings() -> None:
    """Show the Accessibility privacy pane so the permission can be turned on."""
    subprocess.run(["open", _ACCESSIBILITY_URL], check=False)
