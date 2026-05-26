from __future__ import annotations

import math

from autolocalize.features.corners import CornerFeature
from autolocalize.map.grid import CellState, OccupancyGrid


def extract_map_corners(
    grid: OccupancyGrid,
    *,
    min_corner_separation: float = 0.15,
    max_corners: int | None = None,
    tip_offset: float = 0.5,
) -> tuple[CornerFeature, ...]:
    """
    Extract wall-corner features from an occupancy grid.

    Corners are placed at the tip of the wall into free space (not cell center)
  so they align better with LIDAR endpoints.
    """
    res = grid.resolution
    corners: list[CornerFeature] = []

    for gy in range(grid.height):
        for gx in range(grid.width):
            if grid.cell_at(gx, gy) != CellState.OCCUPIED:
                continue

            free_dirs: list[tuple[int, int]] = []
            for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                nx, ny = gx + dx, gy + dy
                if not grid.in_bounds(nx, ny):
                    free_dirs.append((dx, dy))
                elif grid.cell_at(nx, ny) != CellState.OCCUPIED:
                    free_dirs.append((dx, dy))

            if len(free_dirs) != 2:
                continue
            (d1x, d1y), (d2x, d2y) = free_dirs
            if d1x * d2x + d1y * d2y != 0:
                continue

            cx, cy = grid.grid_to_world_center(gx, gy)
            bx, by = d1x + d2x, d1y + d2y
            norm = math.hypot(bx, by)
            if norm < 1e-9:
                continue
            wx = cx + tip_offset * res * bx / norm
            wy = cy + tip_offset * res * by / norm
            angle = math.atan2(by, bx)
            corners.append(
                CornerFeature(
                    x=wx,
                    y=wy,
                    angle=angle,
                    sharpness=math.pi / 2,
                )
            )

    corners = _filter_nearby_corners(corners, min_corner_separation)
    if max_corners is not None and len(corners) > max_corners:
        corners.sort(key=lambda c: c.sharpness, reverse=True)
        corners = corners[:max_corners]
    return tuple(corners)


def _filter_nearby_corners(
    corners: list[CornerFeature], min_separation: float
) -> list[CornerFeature]:
    kept: list[CornerFeature] = []
    for corner in corners:
        if all(corner.distance_to(k) >= min_separation for k in kept):
            kept.append(corner)
    return kept
