"""Webcam capture through OpenCV's AVFoundation backend.

OpenCV does not use AVFoundation's default device order. It concatenates
video devices and muxed devices, then sorts them by `uniqueID`. Camera
indexes passed to `cv2.VideoCapture` follow that sort. `--list-cameras`
uses the same order so index 0 is the device OpenCV will actually open.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from AVFoundation import (
    AVCaptureDevice,
    AVCaptureDeviceTypeBuiltInWideAngleCamera,
    AVMediaTypeMuxed,
    AVMediaTypeVideo,
)

# Requested size only. The device may choose another resolution, and every
# later stage reads the frame's real width and height.
PREFERRED_WIDTH = 1280
PREFERRED_HEIGHT = 720


class CameraError(Exception):
    """Raised when the webcam cannot be listed or opened."""


@dataclass(frozen=True)
class CameraInfo:
    index: int
    name: str
    unique_id: str
    is_builtin: bool

    @property
    def kind(self) -> str:
        return "built-in" if self.is_builtin else "external"


def list_cameras() -> tuple[CameraInfo, ...]:
    """Return cameras in the same order OpenCV's AVFoundation backend uses."""
    devices = _video_devices() + _muxed_devices()
    devices.sort(key=lambda device: str(device.uniqueID()))
    builtin_type = str(AVCaptureDeviceTypeBuiltInWideAngleCamera)
    cameras: list[CameraInfo] = []
    for index, device in enumerate(devices):
        cameras.append(
            CameraInfo(
                index=index,
                name=str(device.localizedName()),
                unique_id=str(device.uniqueID()),
                is_builtin=str(device.deviceType()) == builtin_type,
            )
        )
    return tuple(cameras)


def default_camera_index(cameras: tuple[CameraInfo, ...] | list[CameraInfo]) -> int:
    """Prefer the built-in FaceTime camera over Continuity Camera or a phone."""
    if not cameras:
        raise CameraError("AirOS could not find a webcam.")
    for camera in cameras:
        if camera.is_builtin:
            return camera.index
    return cameras[0].index


def camera_by_index(cameras: tuple[CameraInfo, ...] | list[CameraInfo], index: int) -> CameraInfo:
    for camera in cameras:
        if camera.index == index:
            return camera
    available = ", ".join(str(camera.index) for camera in cameras) or "none"
    raise CameraError(
        "AirOS does not have a camera at index "
        f"{index}. Available indexes: {available}. "
        "Run `airos --list-cameras` to see their names."
    )


class Camera:
    """A single opened webcam. Frames are BGR uint8 arrays from OpenCV."""

    def __init__(
        self,
        info: CameraInfo,
        width: int = PREFERRED_WIDTH,
        height: int = PREFERRED_HEIGHT,
    ) -> None:
        self.info = info
        self._width = width
        self._height = height
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        if self._cap is not None:
            return
        cap = cv2.VideoCapture(self.info.index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap.release()
            raise CameraError(
                f"AirOS could not open {self.info.name} (camera index {self.info.index})."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap

    def read(self) -> np.ndarray | None:
        """Return one BGR frame, or None if the device missed this grab.

        The returned array is only valid until the next `read` call.
        """
        if self._cap is None:
            raise CameraError("AirOS tried to read a frame before opening the camera.")
        ok, frame = self._cap.read()
        if not ok or frame is None or frame.size == 0:
            return None
        return frame

    def close(self) -> None:
        cap = self._cap
        self._cap = None
        if cap is not None:
            cap.release()

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _video_devices() -> list:
    return _device_list(AVCaptureDevice.devicesWithMediaType_(AVMediaTypeVideo))


def _muxed_devices() -> list:
    return _device_list(AVCaptureDevice.devicesWithMediaType_(AVMediaTypeMuxed))


def _device_list(devices) -> list:
    if not devices:
        return []
    return [devices[index] for index in range(len(devices))]
