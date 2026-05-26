"""Point-to-line ICP post-refinement against map wall segments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from autolocalize.geometry.pose import Pose2D
from autolocalize.localization.config import InitialLocalizerConfig
from autolocalize.localization.fast_grid import FastOccupancyLookup
from autolocalize.map.walls import WallSegmentIndex

try:
    from autolocalize._native import WallMapNative
except ImportError:  # pragma: no cover
    WallMapNative = None  # type: ignore[misc, assignment]

if TYPE_CHECKING:
    from autolocalize._native import WallMapNative as WallMapNativeType
else:
    WallMapNativeType = Any


@dataclass(frozen=True, slots=True)
class IcpRefineResult:
    pose: Pose2D
    mean_residual: float
    iterations: int
    converged: bool


def _lookup_wall_index(
    lookup: FastOccupancyLookup,
    *,
    bucket_size: float,
) -> WallSegmentIndex | WallMapNativeType:
    cached = lookup._wall_index.get(bucket_size)
    if cached is not None:
        return cached

    if WallMapNative is not None:
        index = WallMapNative(
            lookup.occupied,
            lookup.origin_x,
            lookup.origin_y,
            lookup.resolution,
            bucket_size,
        )
    else:
        index = WallSegmentIndex.from_lookup(lookup, bucket_size=bucket_size)
    lookup._wall_index[bucket_size] = index
    return index


def refine_pose_icp(
    lookup: FastOccupancyLookup,
    pose: Pose2D,
    local_xy: np.ndarray,
    cfg: InitialLocalizerConfig,
) -> IcpRefineResult:
    """
    Optional post-localization refine: point-to-line ICP with Levenberg–Marquardt.

    Expects a pose already close to the true solution (e.g. after adaptive localize).
    """
    points = np.ascontiguousarray(local_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("local_xy must be an Nx2 array")
    if points.shape[0] < cfg.icp_min_points:
        return IcpRefineResult(pose, 0.0, 0, False)

    wall_index = _lookup_wall_index(
        lookup, bucket_size=cfg.icp_bucket_size
    )

    if WallMapNative is not None and isinstance(wall_index, WallMapNative):
        x, y, theta, mean_residual, iterations, converged = wall_index.refine_icp(
            points,
            pose.x,
            pose.y,
            pose.theta,
            cfg.icp_max_iterations,
            cfg.icp_max_association_dist,
            cfg.icp_convergence_translation,
            cfg.icp_convergence_rotation,
            cfg.icp_huber_delta,
            cfg.icp_min_points,
        )
        return IcpRefineResult(
            Pose2D(x, y, theta),
            float(mean_residual),
            int(iterations),
            bool(converged),
        )

    return _refine_icp_python(
        wall_index,
        pose,
        points,
        cfg,
    )


def _refine_icp_python(
    walls: WallSegmentIndex,
    pose: Pose2D,
    local_xy: np.ndarray,
    cfg: InitialLocalizerConfig,
) -> IcpRefineResult:
    x, y, theta = pose.x, pose.y, pose.theta
    lambda_damp = 1e-3
    last_cost = math.inf
    iterations = 0
    converged = False
    mean_residual = 0.0

    for _ in range(cfg.icp_max_iterations):
        c = math.cos(theta)
        s = math.sin(theta)
        world = np.column_stack(
            (
                x + c * local_xy[:, 0] - s * local_xy[:, 1],
                y + s * local_xy[:, 0] + c * local_xy[:, 1],
            )
        )

        jtj = np.zeros((3, 3), dtype=np.float64)
        jtr = np.zeros(3, dtype=np.float64)
        cost = 0.0
        used = 0

        for i, (wx, wy) in enumerate(world):
            match = walls.nearest_segment(float(wx), float(wy), cfg.icp_max_association_dist)
            if match is None:
                continue
            residual, gx, gy = match
            abs_r = abs(residual)
            weight = (
                1.0
                if abs_r <= cfg.icp_huber_delta or cfg.icp_huber_delta <= 0.0
                else cfg.icp_huber_delta / abs_r
            )
            lx, ly = local_xy[i]
            dwx_dtheta = -s * lx - c * ly
            dwy_dtheta = c * lx - s * ly
            j = np.array([gx, gy, gx * dwx_dtheta + gy * dwy_dtheta], dtype=np.float64)
            jtj += weight * np.outer(j, j)
            jtr += weight * residual * j
            cost += weight * residual * residual
            used += 1

        iterations += 1
        mean_residual = math.sqrt(cost / used) if used else 0.0
        if used < cfg.icp_min_points:
            break

        jtj.flat[0] += lambda_damp
        jtj.flat[4] += lambda_damp
        jtj.flat[8] += lambda_damp
        try:
            delta = np.linalg.solve(jtj, -jtr)
        except np.linalg.LinAlgError:
            break

        if cost >= last_cost:
            lambda_damp = min(lambda_damp * 10.0, 1e6)
            continue

        new_x = x + float(delta[0])
        new_y = y + float(delta[1])
        new_theta = Pose2D.normalize_angle(theta + float(delta[2]))
        x, y, theta = new_x, new_y, new_theta
        lambda_damp = max(lambda_damp * 0.3, 1e-6)
        last_cost = cost

        if (
            math.hypot(float(delta[0]), float(delta[1]))
            <= cfg.icp_convergence_translation
            and abs(float(delta[2])) <= cfg.icp_convergence_rotation
        ):
            converged = True
            break

    return IcpRefineResult(Pose2D(x, y, theta), mean_residual, iterations, converged)
