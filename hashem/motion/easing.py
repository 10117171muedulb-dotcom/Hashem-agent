"""Easing curves.

Every function maps ``t`` in ``[0, 1]`` to a progress value that may overshoot
outside ``[0, 1]`` (back/elastic intentionally do).  All of them are pure and
total: any float in is a float out, so the renderer never has to guard against
``NaN``/domain errors coming from here.
"""

from __future__ import annotations

import math
from typing import Callable

__all__ = ["ease", "EASINGS", "easing_names"]

Easing = Callable[[float], float]


def _clamp01(t: float) -> float:
    if t != t:  # NaN guard
        return 0.0
    return 0.0 if t < 0.0 else (1.0 if t > 1.0 else float(t))


# --------------------------------------------------------------------- linear
def linear(t: float) -> float:
    return _clamp01(t)


# -------------------------------------------------------------- polynomial
def _poly(power: int, mode: str) -> Easing:
    def fn(t: float) -> float:
        x = _clamp01(t)
        if mode == "in":
            return x ** power
        if mode == "out":
            return 1.0 - (1.0 - x) ** power
        # in-out
        if x < 0.5:
            return (2.0 ** (power - 1)) * (x ** power)
        return 1.0 - ((-2.0 * x + 2.0) ** power) / 2.0

    fn.__name__ = f"ease{mode.capitalize()}Pow{power}"
    return fn


def ease_in_quad(t: float) -> float:
    return _poly(2, "in")(t)


def ease_out_quad(t: float) -> float:
    return _poly(2, "out")(t)


def ease_in_out_quad(t: float) -> float:
    return _poly(2, "inOut")(t)


def ease_in_cubic(t: float) -> float:
    return _poly(3, "in")(t)


def ease_out_cubic(t: float) -> float:
    return _poly(3, "out")(t)


def ease_in_out_cubic(t: float) -> float:
    return _poly(3, "inOut")(t)


def ease_in_quart(t: float) -> float:
    return _poly(4, "in")(t)


def ease_out_quart(t: float) -> float:
    return _poly(4, "out")(t)


def ease_in_out_quart(t: float) -> float:
    return _poly(4, "inOut")(t)


def ease_in_quint(t: float) -> float:
    return _poly(5, "in")(t)


def ease_out_quint(t: float) -> float:
    return _poly(5, "out")(t)


def ease_in_out_quint(t: float) -> float:
    return _poly(5, "inOut")(t)


# --------------------------------------------------------------- trigonometric
def ease_in_sine(t: float) -> float:
    x = _clamp01(t)
    return 1.0 - math.cos((x * math.pi) / 2.0)


def ease_out_sine(t: float) -> float:
    x = _clamp01(t)
    return math.sin((x * math.pi) / 2.0)


def ease_in_out_sine(t: float) -> float:
    x = _clamp01(t)
    return -(math.cos(math.pi * x) - 1.0) / 2.0


# ------------------------------------------------------------------ circular
def ease_in_circ(t: float) -> float:
    x = _clamp01(t)
    return 1.0 - math.sqrt(max(0.0, 1.0 - x * x))


def ease_out_circ(t: float) -> float:
    x = _clamp01(t)
    return math.sqrt(max(0.0, 1.0 - (x - 1.0) ** 2))


def ease_in_out_circ(t: float) -> float:
    x = _clamp01(t)
    if x < 0.5:
        return (1.0 - math.sqrt(max(0.0, 1.0 - (2.0 * x) ** 2))) / 2.0
    inner = -2.0 * x + 2.0
    return (math.sqrt(max(0.0, 1.0 - inner * inner)) + 1.0) / 2.0


# -------------------------------------------------------------- exponential
def ease_in_expo(t: float) -> float:
    x = _clamp01(t)
    return 0.0 if x == 0.0 else 2.0 ** (10.0 * x - 10.0)


def ease_out_expo(t: float) -> float:
    x = _clamp01(t)
    return 1.0 if x == 1.0 else 1.0 - 2.0 ** (-10.0 * x)


def ease_in_out_expo(t: float) -> float:
    x = _clamp01(t)
    if x in (0.0, 1.0):
        return x
    if x < 0.5:
        return (2.0 ** (20.0 * x - 10.0)) / 2.0
    return (2.0 - 2.0 ** (-20.0 * x + 10.0)) / 2.0


# ---------------------------------------------------------------------- back
_C1 = 1.70158
_C2 = _C1 * 1.525
_C3 = _C1 + 1.0
_C4 = (2.0 * math.pi) / 3.0
_C5 = (2.0 * math.pi) / 4.5


def ease_in_back(t: float) -> float:
    x = _clamp01(t)
    return _C3 * x ** 3 - _C1 * x ** 2


def ease_out_back(t: float) -> float:
    x = _clamp01(t) - 1.0
    return 1.0 + _C3 * x ** 3 + _C1 * x ** 2


def ease_in_out_back(t: float) -> float:
    x = _clamp01(t)
    if x < 0.5:
        return ((2.0 * x) ** 2 * ((_C2 + 1.0) * 2.0 * x - _C2)) / 2.0
    shifted = 2.0 * x - 2.0
    return (shifted ** 2 * ((_C2 + 1.0) * shifted + _C2) + 2.0) / 2.0


# ------------------------------------------------------------------- elastic
def ease_in_elastic(t: float) -> float:
    x = _clamp01(t)
    if x in (0.0, 1.0):
        return x
    return -(2.0 ** (10.0 * x - 10.0)) * math.sin((x * 10.0 - 10.75) * _C4)


def ease_out_elastic(t: float) -> float:
    x = _clamp01(t)
    if x in (0.0, 1.0):
        return x
    return 2.0 ** (-10.0 * x) * math.sin((x * 10.0 - 0.75) * _C4) + 1.0


def ease_in_out_elastic(t: float) -> float:
    x = _clamp01(t)
    if x in (0.0, 1.0):
        return x
    if x < 0.5:
        return -(2.0 ** (20.0 * x - 10.0) * math.sin((20.0 * x - 11.125) * _C5)) / 2.0
    return (2.0 ** (-20.0 * x + 10.0) * math.sin((20.0 * x - 11.125) * _C5)) / 2.0 + 1.0


# -------------------------------------------------------------------- bounce
def _bounce_out(x: float) -> float:
    n1, d1 = 7.5625, 2.75
    if x < 1.0 / d1:
        return n1 * x * x
    if x < 2.0 / d1:
        x -= 1.5 / d1
        return n1 * x * x + 0.75
    if x < 2.5 / d1:
        x -= 2.25 / d1
        return n1 * x * x + 0.9375
    x -= 2.625 / d1
    return n1 * x * x + 0.984375


def ease_out_bounce(t: float) -> float:
    return _bounce_out(_clamp01(t))


def ease_in_bounce(t: float) -> float:
    return 1.0 - _bounce_out(1.0 - _clamp01(t))


def ease_in_out_bounce(t: float) -> float:
    x = _clamp01(t)
    if x < 0.5:
        return (1.0 - _bounce_out(1.0 - 2.0 * x)) / 2.0
    return (1.0 + _bounce_out(2.0 * x - 1.0)) / 2.0


# --------------------------------------------------------------------- custom
def cubic_bezier(x1: float, y1: float, x2: float, y2: float) -> Easing:
    """Return a CSS-style ``cubic-bezier(x1, y1, x2, y2)`` easing.

    Solved with Newton-Raphson plus a bisection fallback, which stays accurate
    for the extreme control points designers actually reach for.
    """

    def sample(t: float, a: float, b: float) -> float:
        return ((1.0 - 3.0 * b + 3.0 * a) * t + (3.0 * b - 6.0 * a)) * t * t + 3.0 * a * t

    def slope(t: float, a: float, b: float) -> float:
        return 3.0 * (1.0 - 3.0 * b + 3.0 * a) * t * t + 2.0 * (3.0 * b - 6.0 * a) * t + 3.0 * a

    def fn(t: float) -> float:
        x = _clamp01(t)
        guess = x
        for _ in range(8):
            error = sample(guess, x1, x2) - x
            if abs(error) < 1e-6:
                return sample(guess, y1, y2)
            derivative = slope(guess, x1, x2)
            if abs(derivative) < 1e-6:
                break
            guess -= error / derivative
        lo, hi = 0.0, 1.0
        guess = x
        for _ in range(32):
            value = sample(guess, x1, x2)
            if abs(value - x) < 1e-6:
                break
            if value > x:
                hi = guess
            else:
                lo = guess
            guess = (lo + hi) / 2.0
        return sample(guess, y1, y2)

    fn.__name__ = f"cubic_bezier({x1},{y1},{x2},{y2})"
    return fn


# Smooth-step used by the compositor for cross-fades between scenes.
def smootherstep(t: float) -> float:
    x = _clamp01(t)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


EASINGS: dict[str, Easing] = {
    "linear": linear,
    "easeInQuad": ease_in_quad,
    "easeOutQuad": ease_out_quad,
    "easeInOutQuad": ease_in_out_quad,
    "easeInCubic": ease_in_cubic,
    "easeOutCubic": ease_out_cubic,
    "easeInOutCubic": ease_in_out_cubic,
    "easeInQuart": ease_in_quart,
    "easeOutQuart": ease_out_quart,
    "easeInOutQuart": ease_in_out_quart,
    "easeInQuint": ease_in_quint,
    "easeOutQuint": ease_out_quint,
    "easeInOutQuint": ease_in_out_quint,
    "easeInSine": ease_in_sine,
    "easeOutSine": ease_out_sine,
    "easeInOutSine": ease_in_out_sine,
    "easeInCirc": ease_in_circ,
    "easeOutCirc": ease_out_circ,
    "easeInOutCirc": ease_in_out_circ,
    "easeInExpo": ease_in_expo,
    "easeOutExpo": ease_out_expo,
    "easeInOutExpo": ease_in_out_expo,
    "easeInBack": ease_in_back,
    "easeOutBack": ease_out_back,
    "easeInOutBack": ease_in_out_back,
    "easeInElastic": ease_in_elastic,
    "easeOutElastic": ease_out_elastic,
    "easeInOutElastic": ease_in_out_elastic,
    "easeInBounce": ease_in_bounce,
    "easeOutBounce": ease_out_bounce,
    "easeInOutBounce": ease_in_out_bounce,
    "smoothstep": smootherstep,
    # friendly aliases — models and humans both reach for these
    "ease": cubic_bezier(0.25, 0.1, 0.25, 1.0),
    "ease-in": ease_in_cubic,
    "ease-out": ease_out_cubic,
    "ease-in-out": ease_in_out_cubic,
    "pop": ease_out_back,
    "snap": ease_out_expo,
    "bounce": ease_out_bounce,
    "elastic": ease_out_elastic,
}


def easing_names() -> list[str]:
    """Sorted list of every accepted easing name (used by docs and validation)."""
    return sorted(EASINGS)


def ease(name: str | None, t: float, default: str = "easeOutCubic") -> float:
    """Look up ``name`` and apply it; unknown names degrade to ``default``."""
    fn = EASINGS.get((name or default).strip())
    if fn is None:
        fn = EASINGS[default]
    value = fn(t)
    if value != value or value in (float("inf"), float("-inf")):  # pragma: no cover
        return _clamp01(t)
    return float(value)
