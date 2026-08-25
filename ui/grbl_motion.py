from __future__ import annotations


def directed_axis_value(axis: str, distance: float, inverted_feed: bool) -> float:
    """Map scan movement to the machine axis direction."""
    axis = axis.upper()
    if axis == "X":
        return -distance if inverted_feed else distance
    if axis == "Y":
        return distance if inverted_feed else -distance
    return distance


def axis_word(axis: str, distance: float, inverted_feed: bool) -> str:
    value = directed_axis_value(axis, distance, inverted_feed)
    return f"{axis.upper()}{value:g}"
