from __future__ import annotations

import math

import numpy as np

from autolocalize.geometry.pose import Pose2D
from autolocalize.map.grid import CellState, OccupancyGrid


class OccupancyRaster:
    """Fast boolean occupancy lookups for scan-to-map scoring."""

    def __init__(self, grid: OccupancyGrid, *, hit_tolerance: float = 0.08) -> None:
        self.origin_x = grid.origin_x
        self.origin_y = grid.origin_y
        self.resolution = grid.resolution
        self.width = grid.width
        self.height = grid.height

        occupied = np.zeros((grid.height, grid.width), dtype=bool)
        for gy in range(grid.height):
            for gx in range(grid.width):
                if grid.cell_at(gx, gy) == CellState.OCCUPIED:
                    occupied[gy, gx] = True

        radius = max(1, math.ceil(hit_tolerance / grid.resolution))
        self._occupied = _dilate(occupied, radius)

    def score_endpoints(self, world_x: np.ndarray, world_y: np.ndarray) -> float:
        """Fraction of (x, y) points that fall on dilated occupied cells."""
        if world_x.size == 0:
            return 0.0
        gx = np.floor((world_x - self.origin_x) / self.resolution).astype(np.int32)
        gy = np.floor((world_y - self.origin_y) / self.resolution).astype(np.int32)
        valid = (gx >= 0) & (gx < self.width) & (gy >= 0) & (gy < self.height)
        hits = np.zeros(world_x.shape, dtype=bool)
        hits[valid] = self._occupied[gy[valid], gx[valid]]
        return float(np.mean(hits))

    def score_poses(
        self,
        local_xy: np.ndarray,
        poses: list[Pose2D],
    ) -> np.ndarray:
        """
        Score many poses at once.

        local_xy: shape (P, 2) scan endpoints in the robot frame.
        Returns shape (N,) hit fractions in [0, 1].
        """
        if not poses or local_xy.size == 0:
            return np.zeros(len(poses), dtype=np.float64)

        lx = local_xy[:, 0]
        ly = local_xy[:, 1]
        scores = np.empty(len(poses), dtype=np.float64)

        for i, pose in enumerate(poses):
            c = math.cos(pose.theta)
            s = math.sin(pose.theta)
            wx = pose.x + c * lx - s * ly
            wy = pose.y + s * lx + c * ly
            scores[i] = self.score_endpoints(wx, wy)

        return scores


def _dilate(occupied: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return occupied
    out = occupied.copy()
    h, w = occupied.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            shifted = np.zeros_like(occupied)
            y0 = max(0, dy)
            y1 = min(h, h + dy)
            x0 = max(0, dx)
            x1 = min(w, w + dx)
            sy0 = max(0, -dy)
            sy1 = sy0 + (y1 - y0)
            sx0 = max(0, -dx)
            sx1 = sx0 + (x1 - x0)
            shifted[y0:y1, x0:x1] = occupied[sy0:sy1, sx0:sx1]
            out |= shifted
    return out
