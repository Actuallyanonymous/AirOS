import pytest

from airos.camera.camera import CameraError, CameraInfo, camera_by_index, default_camera_index


def _camera(index: int, name: str, unique_id: str, builtin: bool) -> CameraInfo:
    return CameraInfo(index=index, name=name, unique_id=unique_id, is_builtin=builtin)


def test_default_camera_prefers_builtin_even_when_it_is_not_index_zero() -> None:
    cameras = (
        _camera(0, "Phone", "AAA", False),
        _camera(1, "FaceTime HD Camera", "ZZZ", True),
    )
    assert default_camera_index(cameras) == 1


def test_default_camera_uses_the_only_available_device() -> None:
    cameras = (_camera(0, "External", "AAA", False),)
    assert default_camera_index(cameras) == 0


def test_unknown_camera_index_names_the_ones_that_exist() -> None:
    cameras = (_camera(0, "FaceTime", "AAA", True),)
    with pytest.raises(CameraError, match="index 2"):
        camera_by_index(cameras, 2)
