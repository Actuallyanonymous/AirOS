"""Wave detection from the path of the hand.

A wave is one finished stroke. It is labeled when the hand stops, not when
it turns around. The pull-back before the swipe is the wind-up.

That matters for a right hand. A sideways swipe usually lifts a little at
the start. The old check treated that lift as WAVE UP, then ignored the
rest of the swipe and the next one. The stroke is kept whole, and its
direction is the axis that actually traveled farthest.

The pull-back before a wave is not a second wave. Moving down and then up,
without stopping in between, is only WAVE UP. A wave counts when the hand
stops at the end of it. Two waves need a short pause between them.

Coordinates match the preview: x grows to the right, y grows downward.
Up is a decrease in y. Distances are fractions of the frame, not pixels.
"""

from dataclasses import dataclass

from airos.gestures.types import Gesture, GestureReading
from airos.tracking.landmarks import clamp01


@dataclass(frozen=True)
class WaveThresholds:
    """Internal cutoffs. Distances are fractions of the frame."""

    min_displacement: float = 0.07
    min_velocity: float = 0.16
    min_duration: float = 0.12
    max_duration: float = 1.10
    # The winning axis must beat the other by this much, so a diagonal is neither.
    min_dominance: float = 1.15
    # Longest one-axis run divided by the path length. An arc still passes.
    min_straightness: float = 0.42
    # How far the hand must head back before a stroke is over and the next begins.
    reversal_travel: float = 0.04
    start_speed: float = 0.16
    # Net speed. Landmark jitter bounces, so a short window never looks stopped.
    stop_speed: float = 0.15
    stop_hold: float = 0.06
    cooldown: float = 0.65
    smoothing: float = 0.65
    gap_limit: float = 0.22

    def __post_init__(self) -> None:
        if self.min_duration <= 0 or self.max_duration < self.min_duration:
            raise ValueError("wave duration window is invalid")
        if self.min_displacement <= 0 or self.min_velocity <= 0:
            raise ValueError("wave displacement and velocity must be positive")
        if self.min_dominance < 1:
            raise ValueError("min_dominance must be at least 1")
        if not 0 < self.smoothing <= 1:
            raise ValueError("smoothing must be in (0, 1]")


@dataclass(frozen=True)
class _Sample:
    t: float
    x: float
    y: float


class WaveDetector:
    """Turn the motion of one point into at most one wave per stroke."""

    def __init__(self, thresholds: WaveThresholds | None = None) -> None:
        self.thresholds = thresholds or WaveThresholds()
        self._samples: list[_Sample] = []
        self._smooth: tuple[float, float] | None = None
        self._origin: _Sample | None = None
        self._peak: _Sample | None = None
        self._slow_since: float | None = None
        self._pending: GestureReading | None = None
        self._ignore_before = -1.0

    def update(
        self,
        timestamp: float,
        point: tuple[float, float] | None,
    ) -> GestureReading | None:
        """Return a wave when a stroke finishes. Otherwise None."""
        if point is None:
            if self._samples and timestamp - self._samples[-1].t > self.thresholds.gap_limit:
                self._reset()
            return None
        if self._samples and timestamp < self._samples[-1].t:
            self._reset()
        elif self._samples and timestamp - self._samples[-1].t > self.thresholds.gap_limit:
            self._reset()

        self._samples.append(_Sample(timestamp, *self._smoothed(point)))
        cutoff = timestamp - self.thresholds.max_duration - 0.25
        while len(self._samples) > 1 and self._samples[0].t < cutoff:
            self._samples.pop(0)

        if self._origin is None:
            if self._speed() >= self.thresholds.start_speed:
                self._origin = self._motion_start()
                self._peak = self._samples[-1]
                self._slow_since = None
            return None
        return self._track(self._samples[-1])

    def _track(self, current: _Sample) -> GestureReading | None:
        assert self._origin is not None and self._peak is not None
        limits = self.thresholds
        if current.t - self._origin.t > limits.max_duration:
            event = _prefer(self._pending, self._classify(self._origin, self._peak))
            self._close(current.t)
            return event

        dist = _hypot(current.x - self._origin.x, current.y - self._origin.y)
        peak_dist = _hypot(self._peak.x - self._origin.x, self._peak.y - self._origin.y)
        if dist >= peak_dist:
            self._peak = current
        elif peak_dist - dist >= limits.reversal_travel and peak_dist >= limits.min_displacement * 0.75:
            # Turned around without stopping. Remember this leg. A small
            # bounce must not throw the wave away, and a wind-up must not
            # be shown if the next leg is the real swipe.
            self._pending = _prefer(self._pending, self._classify(self._origin, self._peak))
            self._ignore_before = self._peak.t
            self._origin = self._peak
            self._peak = current
            self._slow_since = None
            return None

        if self._net_speed(0.15) < limits.stop_speed:
            if self._slow_since is None:
                self._slow_since = current.t
            elif current.t - self._slow_since >= limits.stop_hold:
                event = _prefer(self._pending, self._classify(self._origin, self._peak))
                self._close(self._peak.t)
                return event
        else:
            self._slow_since = None
        return None

    def _close(self, at_time: float) -> None:
        self._ignore_before = at_time
        self._origin = None
        self._peak = None
        self._slow_since = None
        self._pending = None

    def _reset(self) -> None:
        self._samples.clear()
        self._smooth = None
        self._origin = None
        self._peak = None
        self._slow_since = None
        self._pending = None

    def _smoothed(self, point: tuple[float, float]) -> tuple[float, float]:
        if self._smooth is None or self.thresholds.smoothing == 1:
            self._smooth = point
            return point
        alpha = self.thresholds.smoothing
        sx, sy = self._smooth
        self._smooth = (
            alpha * point[0] + (1.0 - alpha) * sx,
            alpha * point[1] + (1.0 - alpha) * sy,
        )
        return self._smooth

    def _motion_start(self) -> _Sample:
        now = self._samples[-1].t
        target = now - 0.12
        chosen = self._samples[-1]
        for sample in self._samples:
            if sample.t < self._ignore_before:
                continue
            chosen = sample
            if sample.t >= target:
                break
        return chosen

    def _speed(self) -> float:
        """How fast the hand is moving right now, so a swipe can start quickly."""
        return self._net_speed(0.08)

    def _net_speed(self, window: float) -> float:
        """Movement over `window` seconds. Jitter cancels; a real swipe does not."""
        if len(self._samples) < 2:
            return 0.0
        now = self._samples[-1]
        earlier = self._samples[0]
        for sample in self._samples:
            if now.t - sample.t >= window:
                earlier = sample
            else:
                break
        dt = now.t - earlier.t
        if dt <= 1e-4:
            return 0.0
        return _hypot(now.x - earlier.x, now.y - earlier.y) / dt

    def _classify(self, origin: _Sample, peak: _Sample) -> GestureReading | None:
        stroke = [sample for sample in self._samples if origin.t - 1e-4 <= sample.t <= peak.t + 1e-4]
        if len(stroke) < 4:
            return None
        duration = stroke[-1].t - stroke[0].t
        limits = self.thresholds
        if duration < limits.min_duration:
            return None
        run_x = _max_signed_run([sample.x for sample in stroke])
        run_y = _max_signed_run([sample.y for sample in stroke])
        path = 0.0
        for older, newer in zip(stroke, stroke[1:]):
            path += _hypot(newer.x - older.x, newer.y - older.y)
        if path < 1e-6:
            return None
        chosen = _choose_direction(run_x, run_y, limits)
        if chosen is None:
            return None
        gesture, travel = chosen
        if travel / duration < limits.min_velocity:
            return None
        if travel / path < limits.min_straightness:
            return None
        confidence = clamp01(0.45 + 0.55 * min(1.0, travel / (limits.min_displacement * 2.2)))
        return GestureReading(gesture, confidence)


def _prefer(
    earlier: GestureReading | None,
    later: GestureReading | None,
) -> GestureReading | None:
    """Keep the swipe the hand finished, unless that leg was only a bounce."""
    if later is None:
        return earlier
    if earlier is None:
        return later
    if later.confidence + 0.04 >= earlier.confidence:
        return later
    return earlier


def _choose_direction(
    run_x: float,
    run_y: float,
    limits: WaveThresholds,
) -> tuple[Gesture, float] | None:
    """Pick the axis that traveled farther in one direction.

    A right-hand swipe often rises while it moves sideways. Direction follows
    the longer run, so that rise does not become WAVE UP when the sideways
    run is the real stroke.
    """
    hx = abs(run_x)
    hy = abs(run_y)
    if hx >= limits.min_displacement and hx >= hy * limits.min_dominance:
        gesture = Gesture.WAVE_RIGHT if run_x > 0 else Gesture.WAVE_LEFT
        return gesture, hx
    if hy >= limits.min_displacement and hy >= hx * limits.min_dominance:
        gesture = Gesture.WAVE_DOWN if run_y > 0 else Gesture.WAVE_UP
        return gesture, hy
    return None


def _max_signed_run(values: list[float]) -> float:
    """Largest one-direction change. Small jitter does not break the run."""
    best = 0.0
    for start in range(len(values)):
        direction = 0
        for index in range(start + 1, len(values)):
            step = values[index] - values[index - 1]
            if abs(step) <= 0.008:
                pass
            elif direction == 0:
                direction = 1 if step > 0 else -1
            elif step * direction < -0.008:
                break
            delta = values[index] - values[start]
            if abs(delta) > abs(best):
                best = delta
    return best


def _hypot(dx: float, dy: float) -> float:
    return (dx * dx + dy * dy) ** 0.5
