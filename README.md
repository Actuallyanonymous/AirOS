# AirOS

Touchless gesture control for macOS. The built-in webcam watches your hand, and later versions will turn a small set of gestures into ordinary Mac actions: scroll, switch windows, screenshot, play/pause, and move the pointer.

This version opens the camera, finds **21 hand landmarks** on device, and performs a mapped action: scroll, switch apps, screenshot, play/pause, click, and move the pointer.

**AirOS processes camera data locally on your Mac. Webcam frames are never uploaded.**

## What you can run today

```text
Webcam → gesture → Mac action
``` 

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
airos
```

Quit with `q` or `esc`, or close the window.

```bash
airos --list-cameras
airos --camera 1          # use a camera other than the built-in one
airos --no-mirror         # raw camera image instead of a mirror
airos --no-actions        # preview only, no clicks or keystrokes
pytest
```

Controlling the Mac needs Accessibility permission. If it is missing, AirOS says so and opens the Accessibility pane. Use `--no-actions` until you want the Mac to respond.

`airos.json` maps gestures to actions. A fist by itself does not take a screenshot. Show an open palm, then close the hand. Hold an open palm to play or pause, once per hold.

The preview is mirrored by default, so your right hand stays on the right side of the window. Landmark numbers are the `Joint` indexes in `src/airos/tracking/landmarks.py`.

Detections smaller than 8% of the frame are ignored. A picture or a face on the wall can otherwise look a little like a hand, and it is too small to gesture with. `--min-span 0` shows every detection while tuning.

The pose line is geometric. Finger length is compared with the palm, so it does not matter how close you are to the camera or which way the hand is turned. A label appears after the same pose holds for a few frames. A half-bent hand stays `Detected: none`.

A wave is the hand moving in one direction: up, down, left, or right. The pull-back before the swipe is not a second wave, so moving down and then up is only `WAVE UP`. Stop briefly between two swipes if you want both. A small wobble does not count. Waves do not scroll or switch windows yet.

The first launch loads macOS's hand-pose model. That takes a few seconds once. After that, detection is a few milliseconds per frame on Apple Silicon.

## Requirements

- macOS with the Vision framework (macOS 11 or newer)
- A webcam. The built-in camera is preferred when one is present
- Python 3.11 or newer
- Camera permission for the app that launches AirOS (Terminal, Cursor, or Python)

Apple Silicon is the tested target. The same stack (OpenCV, PyObjC, and Vision) is what an Intel Mac would use; that path is not the one exercised in development.

If camera access is off, AirOS stops and tells you:

> AirOS needs Camera permission to see your hand.

Turn it on in **System Settings → Privacy & Security → Camera**.

## How this preview is built

| Piece | Choice | Why |
| --- | --- | --- |
| Camera | OpenCV (`CAP_AVFOUNDATION`) | Small capture loop and a real preview window. Frame size is read from each image, not assumed. |
| Landmarks | Apple Vision `VNDetectHumanHandPoseRequest` via PyObjC | 21 joints, on device, built into macOS, works on Apple Silicon and Intel. No model download and no telemetry. |
| Drawing | OpenCV | The preview is a development view, not the eventual menu-bar app. |

MediaPipe's current macOS wheel is Apple Silicon only, and its Tasks API reports usage metrics to Google. That does not fit a local-only Mac utility, so the landmark detector is Vision.

Camera indexes match OpenCV's AVFoundation backend: video devices, then muxed devices, sorted by `uniqueID`. `--list-cameras` uses that order. The built-in wide camera is the default, so a paired iPhone (Continuity Camera) is not selected just because it appears in the list.

## Architecture

Recognition and actions stay separate, even though only the left half exists today.

```text
Camera
  ↓
Hand tracker          21 landmarks, image coordinates, handedness
  ↓
Gesture engine        pose and wave labels
  ↓
Action engine         scroll, apps, screenshot, pointer
```

Landmarks are normalized (origin top-left, x right, y down). Gesture code will measure distances relative to the hand, and waves relative to the frame, so thresholds do not depend on a particular resolution or how far you sit from the camera.

What the actions do:

- **Wave up / down** scrolls. **Wave left / right** switches to the previous or next app.
- **Open palm, then close the hand** takes a screenshot. A fist by itself does not.
- **Open palm, held** plays or pauses once.
- **Pinch** clicks when you release. **Pinch and move** drags.
- **Pointing** moves the pointer. Small jitters are ignored.

`airos.json` maps those gestures to actions. Accessibility permission is required. Screenshots also need Screen Recording permission for Cursor.

## Privacy

Frames are read from the webcam, passed to Vision in memory, drawn on the preview, and discarded. AirOS does not include a network client, analytics, or a cloud vision API.

## Status

| Phase | State |
| --- | --- |
| 1. Camera | Preview |
| 2. 21 landmarks | Preview |
| 3. Static gestures | Preview |
| 4. Waves | Preview |
| 5. macOS actions | Preview |
| 6. Reliability tuning | Not started |
| 7. Menu bar app | Not started |
| 8. `.app` / `.dmg` | Not started |
