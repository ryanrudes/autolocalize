from __future__ import annotations

import math

from autolocalize.geometry.pose import Pose2D


def rotate_point(x: float, y: float, angle: float) -> tuple[float, float]:
    c = math.cos(angle)
    s = math.sin(angle)
    return c * x - s * y, s * x + c * y


def apply_pose(pose: Pose2D, x: float, y: float) -> tuple[float, float]:
    """Map a point from the robot frame into the world / map frame."""
    rx, ry = rotate_point(x, y, pose.theta)
    return pose.x + rx, pose.y + ry


def apply_pose_to_points(
    pose: Pose2D, points: tuple[tuple[float, float], ...]
) -> tuple[tuple[float, float], ...]:
    return tuple(apply_pose(pose, x, y) for x, y in points)


def pose_from_correspondences(
    src_a: tuple[float, float],
    src_b: tuple[float, float],
    dst_a: tuple[float, float],
    dst_b: tuple[float, float],
) -> Pose2D:
    """
    SE(2) transform mapping src frame into dst (map) frame using two point pairs.
    """
    sx1, sy1 = src_a
    sx2, sy2 = src_b
    dx1, dy1 = dst_a
    dx2, dy2 = dst_b

    src_angle = math.atan2(sy2 - sy1, sx2 - sx1)
    dst_angle = math.atan2(dy2 - dy1, dx2 - dx1)
    theta = Pose2D.normalize_angle(dst_angle - src_angle)

    rotated_x, rotated_y = rotate_point(sx1, sy1, theta)
    return Pose2D(x=dx1 - rotated_x, y=dy1 - rotated_y, theta=theta)


def pose_error(estimated: Pose2D, truth: Pose2D) -> tuple[float, float]:
    """Return (translation error in m, absolute heading error in rad)."""
    dx = estimated.x - truth.x
    dy = estimated.y - truth.y
    dtheta = abs(Pose2D.normalize_angle(estimated.theta - truth.theta))
    return math.hypot(dx, dy), dtheta
