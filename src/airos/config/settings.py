"""Load the gesture-to-action map.

Defaults live in code. A file named airos.json in the current directory, or
next to the project, replaces any keys it contains.
"""

import json
from pathlib import Path

from airos.actions.engine import DEFAULT_MAPPINGS, ActionSettings


def load_settings(path: Path | None = None) -> ActionSettings:
    settings = ActionSettings()
    file = path if path is not None else _find_config()
    if file is None or not file.is_file():
        return settings
    data = json.loads(file.read_text())
    mappings = dict(DEFAULT_MAPPINGS)
    if isinstance(data.get("mappings"), dict):
        mappings.update({str(key): str(value) for key, value in data["mappings"].items()})
    numbers = {
        name: data[name]
        for name in (
            "palm_hold",
            "play_cooldown",
            "close_window",
            "palm_grace",
            "scroll_lines",
            "cursor_smoothing",
            "cursor_deadzone",
            "drag_slop",
            "camera_margin",
        )
        if name in data
    }
    return ActionSettings(mappings=mappings, **numbers)


def _find_config() -> Path | None:
    candidates = (
        Path.cwd() / "airos.json",
        Path(__file__).resolve().parents[3] / "airos.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None
