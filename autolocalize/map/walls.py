"""Occupancy-grid wall segments for scan-to-map ICP."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from autolocalize.localization.fast_grid import FastOccupancyLookup


@dataclass
class WallSegmentIndex:
    """Axis-aligned wall segments extracted from occupied cells."""

    segments: np.ndarray
    bucket_size: float
    buckets: dict[tuple[int, int], list[int]] = field(default_factory=dict)

    @classmethod
    def from_lookup(
        cls, lookup: FastOccupancyLookup, *, bucket_size: float = 0.5
    ) -> WallSegmentIndex:
        occupied = lookup.occupied
        height, width = occupied.shape
        resolution = lookup.resolution
        origin_x = lookup.origin_x
        origin_y = lookup.origin_y
        segments: list[tuple[float, float, float, float]] = []

        for gy in range(height):
            for gx in range(width):
                if not occupied[gy, gx]:
                    continue
                x0 = origin_x + gx * resolution
                y0 = origin_y + gy * resolution
                x1 = x0 + resolution
                y1 = y0 + resolution
                if gy + 1 >= height or not occupied[gy + 1, gx]:
                    segments.append((x0, y1, x1, y1))
                if gy - 1 < 0 or not occupied[gy - 1, gx]:
                    segments.append((x0, y0, x1, y0))
                if gx - 1 < 0 or not occupied[gy, gx - 1]:
                    segments.append((x0, y0, x0, y1))
                if gx + 1 >= width or not occupied[gy, gx + 1]:
                    segments.append((x1, y0, x1, y1))

        index = cls(
            np.asarray(segments, dtype=np.float64),
            bucket_size=max(0.25, bucket_size),
        )
        index._build_buckets()
        return index

    def _build_buckets(self) -> None:
        self.buckets.clear()
        for i, (ax, ay, bx, by) in enumerate(self.segments):
            min_ix = math.floor(min(ax, bx) / self.bucket_size)
            min_iy = math.floor(min(ay, by) / self.bucket_size)
            max_ix = math.floor(max(ax, bx) / self.bucket_size)
            max_iy = math.floor(max(ay, by) / self.bucket_size)
            for iy in range(min_iy, max_iy + 1):
                for ix in range(min_ix, max_ix + 1):
                    self.buckets.setdefault((ix, iy), []).append(i)

    def nearest_segment(
        self, px: float, py: float, max_distance: float
    ) -> tuple[float, float, float] | None:
        if self.segments.size == 0:
            return None

        ix = math.floor(px / self.bucket_size)
        iy = math.floor(py / self.bucket_size)
        radius = max(1, math.ceil(max_distance / self.bucket_size))
        best: tuple[float, float, float] | None = None
        best_dist = max_distance

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                for seg_idx in self.buckets.get((ix + dx, iy + dy), ()):
                    ax, ay, bx, by = self.segments[seg_idx]
                    dist, gx, gy = _point_segment_distance(px, py, ax, ay, bx, by)
                    if dist < best_dist:
                        best_dist = dist
                        best = (dist, gx, gy)
        return best


def _point_segment_distance(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> tuple[float, float, float]:
    vx = bx - ax
    vy = by - ay
    wx = px - ax
    wy = py - ay
    len_sq = vx * vx + vy * vy
    if len_sq <= 1e-12:
        dist = math.hypot(px - ax, py - ay)
        if dist <= 1e-12:
            return dist, 0.0, 0.0
        return dist, (px - ax) / dist, (py - ay) / dist

    t = max(0.0, min(1.0, (wx * vx + wy * vy) / len_sq))
    cx = ax + t * vx
    cy = ay + t * vy
    dx = px - cx
    dy = py - cy
    dist = math.hypot(dx, dy)
    if dist <= 1e-12:
        return 0.0, 0.0, 0.0
    return dist, dx / dist, dy / dist
