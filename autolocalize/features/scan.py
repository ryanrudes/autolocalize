from __future__ import annotations

import math

from autolocalize.features.corners import CornerFeature
from autolocalize.geometry.pose import Pose2D
from autolocalize.sensors.lidar import LidarScan


def scan_to_points(
    scan: LidarScan,
    *,
    range_min: float | None = None,
    range_max: float | None = None,
) -> tuple[tuple[float, float], ...]:
    """Convert a scan to valid endpoints in the robot / LIDAR frame."""
    points: list[tuple[float, float]] = []
    for r, a in zip(scan.ranges, scan.angles):
        if range_min is not None and r <= range_min:
            continue
        if range_max is not None and r >= range_max:
            continue
        points.append((r * math.cos(a), r * math.sin(a)))
    return tuple(points)


def extract_scan_corners(
    scan: LidarScan,
    *,
    range_min: float = 0.05,
    range_max: float | None = None,
    corner_angle_min: float = math.radians(25),
    min_corner_separation: float = 0.15,
    max_corners: int = 20,
) -> tuple[CornerFeature, ...]:
    """
    Extract corner features from a range scan.

    Uses turning angle between consecutive endpoints (works for full 360° scans).
    Returns corners in the robot / LIDAR frame.
    """
    points = list(scan_to_points(scan, range_min=range_min, range_max=range_max))
    if len(points) < 3:
        return ()

    corners = _corners_from_turning_angles(points, corner_angle_min)
    corners = _filter_nearby_corners(corners, min_corner_separation)
    corners.sort(key=lambda c: c.sharpness, reverse=True)
    return tuple(corners[:max_corners])


def _corners_from_turning_angles(
    points: list[tuple[float, float]],
    corner_angle_min: float,
) -> list[CornerFeature]:
    n = len(points)
    if n < 3:
        return []

    corners: list[CornerFeature] = []
    for i in range(n):
        p0 = points[(i - 1) % n]
        p1 = points[i]
        p2 = points[(i + 1) % n]

        v1 = (p1[0] - p0[0], p1[1] - p0[1])
        v2 = (p2[0] - p1[0], p2[1] - p1[1])
        l1 = math.hypot(v1[0], v1[1])
        l2 = math.hypot(v2[0], v2[1])
        if l1 < 1e-6 or l2 < 1e-6:
            continue

        a1 = math.atan2(v1[1], v1[0])
        a2 = math.atan2(v2[1], v2[0])
        turn = Pose2D.normalize_angle(a2 - a1)
        sharpness = abs(turn)

        if sharpness < corner_angle_min:
            continue

        bisector = Pose2D.normalize_angle(a1 + turn / 2)
        corners.append(
            CornerFeature(
                x=p1[0],
                y=p1[1],
                angle=bisector,
                sharpness=sharpness,
            )
        )
    return corners


def _filter_nearby_corners(
    corners: list[CornerFeature], min_separation: float
) -> list[CornerFeature]:
    kept: list[CornerFeature] = []
    for corner in sorted(corners, key=lambda c: c.sharpness, reverse=True):
        if all(corner.distance_to(k) >= min_separation for k in kept):
            kept.append(corner)
    return kept
