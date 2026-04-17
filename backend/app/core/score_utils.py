from __future__ import annotations


def to_percentage(score: float | int | None) -> float:
    """Normalize score values to a 0-100 percentage.

    Supports legacy mixed scales:
    - 0.0..1.0  -> *100
    - 0.0..10.0 -> *10
    - 0.0..100.0 -> as-is
    """
    if score is None:
        return 0.0

    value = float(score)
    if value <= 1.0:
        value *= 100.0
    elif value <= 10.0:
        value *= 10.0

    return max(0.0, min(100.0, value))
