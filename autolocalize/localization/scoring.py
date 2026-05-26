from __future__ import annotations

import math

from autolocalize.features.corners import CornerFeature
from autolocalize.features.scan import scan_to_points
from autolocalize.geometry.pose import Pose2D
from autolocalize.geometry.transform import apply_pose
from autolocalize.map.grid import CellState, OccupancyGrid
from autolocalize.sensors.lidar import LidarScan


def score_scan_against_map(
    grid: OccupancyGrid,
    scan: LidarScan,
    pose: Pose2D,
    *,
    range_min: float = 0.05,
    range_max: float | None = None,
    hit_tolerance: float = 0.08,
    ray_stride: int = 1,
) -> float:
    """
    Score how well scan endpoints align with map obstacles after applying pose.

    Returns a value in [0, 1]: fraction of scan points landing near occupied cells.
    """
    points = scan_to_points(scan, range_min=range_min, range_max=range_max)
    if not points:
        return 0.0

    tol_cells = max(1, math.ceil(hit_tolerance / grid.resolution))
    hits = 0

    stride = max(1, ray_stride)
    sampled = points[::stride]
    for lx, ly in sampled:
        wx, wy = apply_pose(pose, lx, ly)
        if _point_near_occupied(grid, wx, wy, tol_cells):
            hits += 1

    return hits / len(sampled)


def score_corner_alignment(
    scan_corners: tuple[CornerFeature, ...],
    map_corners: tuple[CornerFeature, ...],
    pose: Pose2D,
    *,
    position_tolerance: float = 0.18,
    angle_tolerance: float = 0.6,
) -> float:
    """
    Fraction of scan corners that land on a matching map corner (position + angle).
    """
    if not scan_corners:
        return 0.0

    matches = 0
    for sc in scan_corners:
        wx, wy = apply_pose(pose, sc.x, sc.y)
        world_bisector = Pose2D.normalize_angle(pose.theta + sc.angle)
        for mc in map_corners:
            if math.hypot(wx - mc.x, wy - mc.y) > position_tolerance:
                continue
            if abs(Pose2D.normalize_angle(world_bisector - mc.angle)) > angle_tolerance:
                continue
            matches += 1
            break

    return matches / len(scan_corners)


def combined_match_score(
    grid: OccupancyGrid,
    scan: LidarScan,
    pose: Pose2D,
    scan_corners: tuple[CornerFeature, ...],
    map_corners: tuple[CornerFeature, ...],
    *,
    range_min: float = 0.05,
    range_max: float | None = None,
    hit_tolerance: float = 0.08,
    corner_weight: float = 0.12,
    ray_stride: int = 1,
) -> float:
    """Endpoint alignment with a corner-consistency tiebreaker."""
    endpoint = score_scan_against_map(
        grid,
        scan,
        pose,
        range_min=range_min,
        range_max=range_max,
        hit_tolerance=hit_tolerance,
        ray_stride=ray_stride,
    )
    corners = score_corner_alignment(scan_corners, map_corners, pose)
    return endpoint + corner_weight * corners


def _point_near_occupied(
    grid: OccupancyGrid, x: float, y: float, radius_cells: int
) -> bool:
    gx, gy = grid.world_to_grid(x, y)
    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            nx, ny = gx + dx, gy + dy
            if not grid.in_bounds(nx, ny):
                continue
            if grid.cell_at(nx, ny) == CellState.OCCUPIED:
                return True
    return False
