from __future__ import annotations

import math
from dataclasses import dataclass

from autolocalize.geometry.pose import Pose2D
from autolocalize.map.grid import OccupancyGrid
from autolocalize.sensors.raycast import cast_ray


@dataclass(frozen=True, slots=True)
class LidarConfig:
    """2D LIDAR configuration."""

    num_rays: int = 720
    angle_min: float = -math.pi
    angle_max: float = math.pi
    range_min: float = 0.05
    range_max: float = 8.0

    def __post_init__(self) -> None:
        if self.num_rays < 1:
            raise ValueError("num_rays must be at least 1")
        if self.range_min < 0:
            raise ValueError("range_min must be non-negative")
        if self.range_max <= self.range_min:
            raise ValueError("range_max must be greater than range_min")
        if self.angle_max <= self.angle_min and self.num_rays > 1:
            raise ValueError("angle_max must be greater than angle_min")

    @property
    def angle_increment(self) -> float:
        if self.num_rays == 1:
            return 0.0
        return (self.angle_max - self.angle_min) / (self.num_rays - 1)

    def beam_angles(self) -> tuple[float, ...]:
        if self.num_rays == 1:
            return ((self.angle_min + self.angle_max) / 2,)
        inc = self.angle_increment
        return tuple(self.angle_min + i * inc for i in range(self.num_rays))


@dataclass(frozen=True, slots=True)
class LidarScan:
    """Simulated LIDAR scan at a pose."""

    pose: Pose2D
    ranges: tuple[float, ...]
    angles: tuple[float, ...]

    @property
    def num_rays(self) -> int:
        return len(self.ranges)


class LidarSimulator:
    """Ray-cast LIDAR against an occupancy grid."""

    def __init__(self, grid: OccupancyGrid, config: LidarConfig | None = None) -> None:
        self.grid = grid
        self.config = config or LidarConfig()

    def scan(self, pose: Pose2D) -> LidarScan:
        """Simulate a full LIDAR sweep at the given world pose."""
        cfg = self.config
        local_angles = cfg.beam_angles()
        ranges: list[float] = []

        for local_angle in local_angles:
            world_angle = pose.rotate(local_angle)
            distance = cast_ray(
                self.grid,
                pose.x,
                pose.y,
                world_angle,
                cfg.range_min,
                cfg.range_max,
            )
            ranges.append(distance)

        return LidarScan(
            pose=pose,
            ranges=tuple(ranges),
            angles=local_angles,
        )

    def ranges_at(self, pose: Pose2D) -> tuple[float, ...]:
        """Return only the range readings (meters) for a pose."""
        return self.scan(pose).ranges
