from __future__ import annotations

import itertools
import math

from autolocalize.features.corners import CornerFeature
from autolocalize.geometry.pose import Pose2D
from autolocalize.geometry.transform import pose_from_correspondences, rotate_point


def generate_feature_hypotheses(
    scan_corners: tuple[CornerFeature, ...],
    map_corners: tuple[CornerFeature, ...],
    *,
    grid_resolution: float = 0.05,
    distance_tolerance: float | None = None,
    try_heading_flip: bool = True,
) -> tuple[list[Pose2D], frozenset[tuple[int, int, int]]]:
    """
    Generate pose hypotheses from scan-to-map corner correspondences.

    Enumerates every scan corner against every map corner (with optional pi flip
    on heading), plus consistent pairs matched by edge length.

    Returns poses and a set of keys for poses derived from corner pairs (used
    to break heading ties when endpoint scores are ambiguous).
    """
    if not scan_corners or not map_corners:
        return [], frozenset()

    if distance_tolerance is None:
        distance_tolerance = max(0.12, grid_resolution * 4.0)

    hypotheses: list[Pose2D] = []
    seen: set[tuple[int, int, int]] = set()
    pair_keys: set[tuple[int, int, int]] = set()

    def pose_key(pose: Pose2D) -> tuple[int, int, int]:
        return (
            int(round(pose.x * 20)),
            int(round(pose.y * 20)),
            int(round(pose.theta * 20)),
        )

    def add_pose(pose: Pose2D, *, from_pair: bool = False) -> None:
        key = pose_key(pose)
        if key not in seen:
            seen.add(key)
            hypotheses.append(pose)
            if from_pair:
                pair_keys.add(key)

    for sc in scan_corners:
        for mc in map_corners:
            for pose in _poses_from_corner_pair(sc, mc, try_heading_flip):
                add_pose(pose)

    if len(scan_corners) >= 2 and len(map_corners) >= 2:
        for (s1, s2), (m1, m2) in itertools.product(
            itertools.combinations(scan_corners, 2),
            itertools.combinations(map_corners, 2),
        ):
            d_scan = s1.distance_to(s2)
            d_map = m1.distance_to(m2)
            if d_scan < 0.15 or d_map < 0.15:
                continue
            if abs(d_scan - d_map) > distance_tolerance:
                continue

            for ma, mb in ((m1, m2), (m2, m1)):
                add_pose(
                    pose_from_correspondences(
                        (s1.x, s1.y),
                        (s2.x, s2.y),
                        (ma.x, ma.y),
                        (mb.x, mb.y),
                    ),
                    from_pair=True,
                )

    return hypotheses, frozenset(pair_keys)


def _poses_from_corner_pair(
    sc: CornerFeature,
    mc: CornerFeature,
    try_heading_flip: bool,
) -> list[Pose2D]:
    """Poses that align one scan corner onto one map corner."""
    theta = Pose2D.normalize_angle(mc.angle - sc.angle)
    poses = [_pose_align_corner(sc, mc, theta)]
    if try_heading_flip:
        poses.append(_pose_align_corner(sc, mc, Pose2D.normalize_angle(theta + math.pi)))
    return poses


def _pose_align_corner(
    sc: CornerFeature, mc: CornerFeature, theta: float
) -> Pose2D:
    rx, ry = rotate_point(sc.x, sc.y, theta)
    return Pose2D(x=mc.x - rx, y=mc.y - ry, theta=theta)


# Kept for optional coarse fallback / tests
def generate_grid_hypotheses(
    grid,
    *,
    xy_step: float = 0.35,
    theta_step: float = math.pi / 8,
    margin: float = 0.15,
    max_poses: int = 2000,
) -> list[Pose2D]:
    """Coarse grid over free space (optional fallback, not used by default)."""
    from autolocalize.map.grid import CellState, OccupancyGrid

    if not isinstance(grid, OccupancyGrid):
        raise TypeError("grid must be OccupancyGrid")

    x_min = grid.origin_x + margin
    y_min = grid.origin_y + margin
    x_max = grid.origin_x + grid.world_width - margin
    y_max = grid.origin_y + grid.world_height - margin

    xy_positions: list[tuple[float, float]] = []
    x = x_min
    while x <= x_max:
        y = y_min
        while y <= y_max:
            gx, gy = grid.world_to_grid(x, y)
            if grid.in_bounds(gx, gy) and grid.cell_at(gx, gy) == CellState.FREE:
                xy_positions.append((x, y))
            y += xy_step
        x += xy_step

    if not xy_positions:
        return []

    thetas = _theta_samples(theta_step)
    if len(xy_positions) * len(thetas) > max_poses:
        stride = max(1, math.ceil(len(xy_positions) / max(1, max_poses // len(thetas))))
        xy_positions = xy_positions[::stride]

    poses: list[Pose2D] = []
    for x, y in xy_positions:
        for theta in thetas:
            poses.append(Pose2D(x, y, theta))
            if len(poses) >= max_poses:
                return poses
    return poses


def _theta_samples(theta_step: float) -> list[float]:
    thetas: list[float] = []
    theta = -math.pi
    while theta < math.pi - theta_step * 0.5:
        thetas.append(theta)
        theta += theta_step
    return thetas
