from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CornerFeature:
    """A wall corner: position and outward bisector angle into free space."""

    x: float
    y: float
    angle: float
    sharpness: float = 1.0

    def distance_to(self, other: CornerFeature) -> float:
        return math.hypot(self.x - other.x, self.y - other.y)
