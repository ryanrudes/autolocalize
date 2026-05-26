from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from autolocalize.features.corners import CornerFeature
from autolocalize.geometry.pose import Pose2D
from autolocalize.map.freespace import FreeSpaceIndex, binary_dilate
from autolocalize.map.grid import CellState, OccupancyGrid

try:
    from autolocalize._native import PoseScorerNative
except ImportError:  # pragma: no cover - editable installs always build native
    PoseScorerNative = None  # type: ignore[misc, assignment]

if TYPE_CHECKING:
    from autolocalize._native import PoseScorerNative as PoseScorerNativeType
else:
    PoseScorerNativeType = Any


class FastOccupancyLookup:
    """Boolean occupancy raster with vectorized endpoint scoring."""

    def __init__(self, grid: OccupancyGrid, *, freespace_noise_cells: int = 2) -> None:
        self.resolution = grid.resolution
        self.origin_x = grid.origin_x
        self.origin_y = grid.origin_y
        self.width = grid.width
        self.height = grid.height
        occupied = np.fromiter(
            (c == CellState.OCCUPIED for c in grid.cells),
            dtype=bool,
            count=grid.width * grid.height,
        ).reshape(grid.height, grid.width)
        self.occupied = occupied
        self.freespace = FreeSpaceIndex(
            grid, noise_cells=freespace_noise_cells
        )
        self._hit_masks: dict[int, np.ndarray] = {}
        self._wall_index: dict[float, object] = {}

    def hit_mask(self, hit_radius_cells: int) -> np.ndarray:
        radius = max(0, hit_radius_cells)
        cached = self._hit_masks.get(radius)
        if cached is None:
            cached = binary_dilate(self.occupied, radius)
            self._hit_masks[radius] = cached
        return cached


def transform_points(pose: Pose2D, local_xy: np.ndarray) -> np.ndarray:
    """Apply pose to Nx2 points."""
    c = math.cos(pose.theta)
    s = math.sin(pose.theta)
    wx = pose.x + c * local_xy[:, 0] - s * local_xy[:, 1]
    wy = pose.y + s * local_xy[:, 0] + c * local_xy[:, 1]
    return np.column_stack((wx, wy))


def world_to_grid(
    lookup: FastOccupancyLookup, wx: np.ndarray, wy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    gx = np.floor((wx - lookup.origin_x) / lookup.resolution).astype(np.int32)
    gy = np.floor((wy - lookup.origin_y) / lookup.resolution).astype(np.int32)
    return gx, gy


def _xy2_array(corners: tuple[CornerFeature, ...]) -> np.ndarray:
    if not corners:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray([(c.x, c.y) for c in corners], dtype=np.float64)


def _angles_array(corners: tuple[CornerFeature, ...]) -> np.ndarray:
    if not corners:
        return np.empty(0, dtype=np.float64)
    return np.asarray([c.angle for c in corners], dtype=np.float64)


def _build_native_scorer(
    lookup: FastOccupancyLookup,
    local_xy: np.ndarray,
    scan_corners: tuple[CornerFeature, ...],
    map_corners: tuple[CornerFeature, ...],
    *,
    hit_mask: np.ndarray,
    position_tolerance: float,
    angle_tolerance: float,
    corner_match_requires_angle: bool,
    freespace_consistency: bool,
    reject_robot_outside_free: bool,
) -> PoseScorerNativeType | None:
    if PoseScorerNative is None:
        return None

    component_labels = None
    reachable_masks: list[np.ndarray] = []
    if freespace_consistency:
        component_labels = lookup.freespace.component_labels
        reachable_masks = [
            lookup.freespace.reachable_region(component_id)
            for component_id in range(lookup.freespace.num_components)
        ]

    return PoseScorerNative(
        hit_mask,
        np.ascontiguousarray(local_xy, dtype=np.float64),
        _xy2_array(scan_corners),
        _angles_array(scan_corners),
        _xy2_array(map_corners),
        _angles_array(map_corners),
        lookup.origin_x,
        lookup.origin_y,
        lookup.resolution,
        position_tolerance * position_tolerance,
        angle_tolerance,
        corner_match_requires_angle,
        freespace_consistency,
        reject_robot_outside_free,
        component_labels,
        reachable_masks,
    )


class PoseScorer:
    """
    Cached scan data for fast repeated pose scoring during localization.

    Any transformed endpoint farther than ``freespace_noise_cells`` outside
    the robot's connected free region makes the pose impossible (score 0).
    Wall hits on occupied cells are always allowed.
    """

    def __init__(
        self,
        lookup: FastOccupancyLookup,
        local_xy: np.ndarray,
        scan_corners: tuple[CornerFeature, ...],
        map_corners: tuple[CornerFeature, ...],
        *,
        hit_radius_cells: int,
        corner_weight: float = 0.12,
        position_tolerance: float = 0.18,
        angle_tolerance: float = 0.6,
        corner_match_requires_angle: bool = False,
        freespace_consistency: bool = True,
        reject_robot_outside_free: bool = True,
    ) -> None:
        self.lookup = lookup
        self.local_xy = local_xy
        self._hit_mask = lookup.hit_mask(hit_radius_cells)
        self.corner_weight = corner_weight
        self.angle_tolerance = angle_tolerance
        self.corner_match_requires_angle = corner_match_requires_angle
        self.freespace_consistency = freespace_consistency
        self.reject_robot_outside_free = reject_robot_outside_free

        self._scan_xy = np.asarray(
            [(c.x, c.y) for c in scan_corners], dtype=np.float64
        )
        self._scan_angles = np.asarray(
            [c.angle for c in scan_corners], dtype=np.float64
        )
        self._map_xy = np.asarray(
            [(c.x, c.y) for c in map_corners], dtype=np.float64
        )
        self._map_angles = np.asarray(
            [c.angle for c in map_corners], dtype=np.float64
        )
        self._pos_tol_sq = position_tolerance * position_tolerance
        self._native = _build_native_scorer(
            lookup,
            local_xy,
            scan_corners,
            map_corners,
            hit_mask=self._hit_mask,
            position_tolerance=position_tolerance,
            angle_tolerance=angle_tolerance,
            corner_match_requires_angle=corner_match_requires_angle,
            freespace_consistency=freespace_consistency,
            reject_robot_outside_free=reject_robot_outside_free,
        )

    @property
    def uses_native(self) -> bool:
        return self._native is not None

    def score_fast(self, pose: Pose2D) -> float:
        if self._native is not None:
            return float(
                self._native.score_fast(pose.x, pose.y, pose.theta)
            )
        return self._score_endpoints(pose)

    def rank_pose(
        self,
        pose: Pose2D,
        *,
        corner_weight: float,
        min_ep_for_corners: float,
    ) -> float:
        if self._native is not None:
            return float(
                self._native.rank_pose(
                    pose.x,
                    pose.y,
                    pose.theta,
                    corner_weight,
                    min_ep_for_corners,
                )
            )
        endpoint = self.score_fast(pose)
        if endpoint < min_ep_for_corners:
            return endpoint
        if corner_weight > 0.0 and self._scan_xy.shape[0] > 0:
            return endpoint + corner_weight * self.score_corners(pose)
        return endpoint

    def _endpoint_plausible(
        self, gx: np.ndarray, gy: np.ndarray, reachable: np.ndarray
    ) -> np.ndarray:
        """True when endpoint is a wall hit or within noise of connected free space."""
        w, h = self.lookup.width, self.lookup.height
        n = gx.shape[0]
        plausible = np.zeros(n, dtype=bool)

        in_bounds = (gx >= 0) & (gx < w) & (gy >= 0) & (gy < h)
        if np.any(in_bounds):
            ix = gx[in_bounds]
            iy = gy[in_bounds]
            plausible[in_bounds] = self._hit_mask[iy, ix] | reachable[iy, ix]

        out_of_bounds = ~in_bounds
        if np.any(out_of_bounds):
            cx = np.clip(gx[out_of_bounds], 0, w - 1)
            cy = np.clip(gy[out_of_bounds], 0, h - 1)
            plausible[out_of_bounds] = reachable[cy, cx]
        return plausible

    def _score_endpoints(self, pose: Pose2D) -> float:
        n = self.local_xy.shape[0]
        if n == 0:
            return 0.0

        c = math.cos(pose.theta)
        s = math.sin(pose.theta)
        wx = pose.x + c * self.local_xy[:, 0] - s * self.local_xy[:, 1]
        wy = pose.y + s * self.local_xy[:, 0] + c * self.local_xy[:, 1]

        lookup = self.lookup
        w, h = lookup.width, lookup.height
        gx, gy = world_to_grid(lookup, wx, wy)

        if self.freespace_consistency:
            robot_gx = int(
                math.floor((pose.x - lookup.origin_x) / lookup.resolution)
            )
            robot_gy = int(
                math.floor((pose.y - lookup.origin_y) / lookup.resolution)
            )
            robot_component = lookup.freespace.robot_component(robot_gx, robot_gy)
            if robot_component is None:
                if self.reject_robot_outside_free:
                    return 0.0
            else:
                reachable = lookup.freespace.reachable_region(robot_component)
                if not np.all(self._endpoint_plausible(gx, gy, reachable)):
                    return 0.0

        in_bounds = (gx >= 0) & (gx < w) & (gy >= 0) & (gy < h)
        hits = int(np.count_nonzero(self._hit_mask[gy[in_bounds], gx[in_bounds]]))
        return hits / n

    def score_corners(self, pose: Pose2D) -> float:
        if self._native is not None:
            return float(
                self._native.score_corners(pose.x, pose.y, pose.theta)
            )
        n_scan = self._scan_xy.shape[0]
        if n_scan == 0 or self._map_xy.shape[0] == 0:
            return 0.0
        world = transform_points(pose, self._scan_xy)
        world_angles = np.remainder(
            pose.theta + self._scan_angles + math.pi, math.tau
        ) - math.pi
        diff = world[:, None, :] - self._map_xy[None, :, :]
        dist_sq = diff[..., 0] ** 2 + diff[..., 1] ** 2
        angle_diff = np.abs(
            np.remainder(
                world_angles[:, None] - self._map_angles[None, :] + math.pi,
                math.tau,
            )
            - math.pi
        )
        if self.corner_match_requires_angle:
            matched = np.any(
                (dist_sq <= self._pos_tol_sq) & (angle_diff <= self.angle_tolerance),
                axis=1,
            )
        else:
            matched = np.any(dist_sq <= self._pos_tol_sq, axis=1)
        return float(np.count_nonzero(matched)) / n_scan

    def corner_assignment_cost(self, pose: Pose2D) -> float:
        """
        Greedy one-to-one scan-to-map corner distance (lower is better).

        Penalizes permutations that swap walls compared to independent nearest-neighbor matching.
        """
        if self._native is not None:
            return float(
                self._native.corner_assignment_cost(pose.x, pose.y, pose.theta)
            )
        n_scan = self._scan_xy.shape[0]
        n_map = self._map_xy.shape[0]
        if n_scan == 0 or n_map == 0:
            return float("inf")
        world = transform_points(pose, self._scan_xy)
        diff = world[:, None, :] - self._map_xy[None, :, :]
        dist = np.sqrt(diff[..., 0] ** 2 + diff[..., 1] ** 2)
        used_map: set[int] = set()
        total = 0.0
        for scan_idx in np.argsort(np.min(dist, axis=1)):
            best_j = None
            best_d = float("inf")
            for j in range(n_map):
                if j in used_map:
                    continue
                if dist[scan_idx, j] < best_d:
                    best_d = dist[scan_idx, j]
                    best_j = j
            if best_j is None:
                return float("inf")
            used_map.add(best_j)
            total += best_d
        return total / n_scan

    def score_full(self, pose: Pose2D) -> float:
        return self.score_fast(pose) + self.corner_weight * self.score_corners(pose)

    def freespace_violation_rate(self, pose: Pose2D) -> float:
        """Fraction of endpoints outside the noise-expanded free region (and not on walls)."""
        if self._native is not None:
            return float(
                self._native.freespace_violation_rate(pose.x, pose.y, pose.theta)
            )
        n = self.local_xy.shape[0]
        if n == 0:
            return 1.0

        c = math.cos(pose.theta)
        s = math.sin(pose.theta)
        wx = pose.x + c * self.local_xy[:, 0] - s * self.local_xy[:, 1]
        wy = pose.y + s * self.local_xy[:, 0] + c * self.local_xy[:, 1]
        gx, gy = world_to_grid(self.lookup, wx, wy)

        robot_gx = int(
            math.floor((pose.x - self.lookup.origin_x) / self.lookup.resolution)
        )
        robot_gy = int(
            math.floor((pose.y - self.lookup.origin_y) / self.lookup.resolution)
        )
        robot_component = self.lookup.freespace.robot_component(robot_gx, robot_gy)
        if robot_component is None:
            return 1.0

        reachable = self.lookup.freespace.reachable_region(robot_component)
        plausible = self._endpoint_plausible(gx, gy, reachable)
        return float(np.count_nonzero(~plausible)) / n
