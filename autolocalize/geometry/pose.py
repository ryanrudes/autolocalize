from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Pose2D:
    """Robot pose in the map / world frame (meters, radians)."""

    x: float
    y: float
    theta: float = 0.0

    def rotate(self, local_angle: float) -> float:
        """Convert a bearing in the robot frame to world-frame angle."""
        return self.theta + local_angle

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Wrap angle to [-pi, pi)."""
        return math.remainder(angle, math.tau)
