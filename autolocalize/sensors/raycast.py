from __future__ import annotations

import math

from autolocalize.map.grid import OccupancyGrid


def cast_ray(
    grid: OccupancyGrid,
    x0: float,
    y0: float,
    angle: float,
    range_min: float,
    range_max: float,
) -> float:
    """
    Cast a single ray and return the reported range in meters.

    Uses grid traversal (Amanatides & Woo). Out-of-map and no-hit within
    range_max both return range_max. Hits closer than range_min are clamped.
    """
    if range_max <= 0:
        return range_max

    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    gx, gy = grid.world_to_grid(x0, y0)
    if not grid.in_bounds(gx, gy) or grid.is_blocking(gx, gy):
        return range_min

    step_x = 1 if cos_a >= 0 else -1
    step_y = 1 if sin_a >= 0 else -1

    if abs(cos_a) < 1e-12:
        t_delta_x = float("inf")
        t_max_x = float("inf")
    else:
        next_boundary_x = (
            (gx + (1 if cos_a >= 0 else 0)) * grid.resolution + grid.origin_x
        )
        t_delta_x = abs(grid.resolution / cos_a)
        t_max_x = abs((next_boundary_x - x0) / cos_a)

    if abs(sin_a) < 1e-12:
        t_delta_y = float("inf")
        t_max_y = float("inf")
    else:
        next_boundary_y = (
            (gy + (1 if sin_a >= 0 else 0)) * grid.resolution + grid.origin_y
        )
        t_delta_y = abs(grid.resolution / sin_a)
        t_max_y = abs((next_boundary_y - y0) / sin_a)

    distance = 0.0
    max_steps = int(range_max / grid.resolution) + grid.width + grid.height + 2

    for _ in range(max_steps):
        if t_max_x < t_max_y:
            distance = t_max_x
            t_max_x += t_delta_x
            gx += step_x
        else:
            distance = t_max_y
            t_max_y += t_delta_y
            gy += step_y

        if distance > range_max:
            return range_max

        if not grid.in_bounds(gx, gy):
            return range_max

        if grid.is_blocking(gx, gy):
            hit = max(distance, range_min)
            return min(hit, range_max)

    return range_max
