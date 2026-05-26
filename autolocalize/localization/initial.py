from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

from autolocalize.features.corners import CornerFeature
from autolocalize.features.map_features import extract_map_corners
from autolocalize.features.scan import extract_scan_corners, scan_to_points
from autolocalize.geometry.pose import Pose2D
from autolocalize.localization.fast_grid import FastOccupancyLookup, PoseScorer
from autolocalize.localization.greedy import greedy_localize
from autolocalize.localization.hypotheses import generate_grid_hypotheses
from autolocalize.localization.refine import refine_pose, refine_pose_multiscale
from autolocalize.map.grid import OccupancyGrid
from autolocalize.sensors.lidar import LidarConfig, LidarScan


@dataclass(frozen=True, slots=True)
class InitialLocalizerConfig:
    """Parameters for feature-based initial localization."""

    min_match_score: float = 0.35
    max_scan_corners: int = 12
    corner_angle_min: float = math.radians(25)
    min_corner_separation: float = 0.12
    hit_tolerance: float = 0.08
    refine_poses: bool = True
    refine_top_k: int = 8
    translation_span: float = 0.2
    rotation_span: float = 0.35
    try_heading_flip: bool = True
    score_ray_stride: int = 6
    final_score_ray_stride: int = 2
    search_corner_weight: float = 0.15
    max_scan_corners_for_pairs: int = 12
    early_exit_score: float | None = None
    freespace_consistency: bool = True
    freespace_noise_margin: float | None = None
    reject_robot_outside_free: bool = True
    use_grid_search: bool = False
    grid_search_on_failure: bool = True
    grid_search_endpoint_threshold: float = 0.85
    grid_xy_step: float = 0.2
    grid_theta_step: float = math.pi / 8
    max_grid_hypotheses: int = 5000
    min_endpoint_for_corner_rank: float = 0.25
    refine_multiscale: bool = True


@dataclass(frozen=True, slots=True)
class LocalizationResult:
    """Outcome of initial localization."""

    pose: Pose2D | None
    score: float
    scan_corners: tuple[CornerFeature, ...]
    map_corners: tuple[CornerFeature, ...]
    hypotheses_tested: int
    stopped_early: bool = False

    @property
    def success(self) -> bool:
        return self.pose is not None


def _pick_best_candidate(
    candidates: list[tuple[float, float, float, Pose2D]],
    *,
    min_match_score: float,
    strong_endpoint: float = 0.92,
) -> tuple[float, float, float, Pose2D]:
    """
    Choose a pose from refined candidates.

    Prefer high endpoint match scores; among ties use lowest corner assignment cost.
    """
    ranked = sorted(candidates, key=lambda item: (-item[0], item[1]))
    best_score, _, best_search_ep, best_pose = ranked[0]
    strong = [item for item in ranked if item[2] >= strong_endpoint]
    if len(strong) >= 2:
        strong.sort(key=lambda item: (item[1], -item[0]))
        best_score, _, best_search_ep, best_pose = strong[0]
    elif best_score < min_match_score and len(ranked) > 1:
        for score, cost, search_ep, pose in ranked[1:]:
            if score >= min_match_score:
                return score, cost, search_ep, pose
    return best_score, 0.0, best_search_ep, best_pose


class InitialLocalizer:
    """
    Estimate the robot pose (LIDAR-to-map transform) from a single scan.

    Map corners are precomputed once. High-value feature correspondences are
    tried first; the top-K hypotheses are refined and ranked with denser scoring.
    """

    def __init__(
        self,
        grid: OccupancyGrid,
        config: InitialLocalizerConfig | None = None,
        *,
        map_corners: tuple[CornerFeature, ...] | None = None,
    ) -> None:
        self.grid = grid
        self.config = config or InitialLocalizerConfig()
        self._map_corners = map_corners
        self._lookup: FastOccupancyLookup | None = None
        self._lookup_noise_cells: int | None = None

    def _noise_cells(self, cfg: InitialLocalizerConfig) -> int:
        noise_margin = (
            cfg.freespace_noise_margin
            if cfg.freespace_noise_margin is not None
            else cfg.hit_tolerance
        )
        return max(1, math.ceil(noise_margin / self.grid.resolution))

    @property
    def lookup(self) -> FastOccupancyLookup:
        noise_cells = self._lookup_noise_cells
        if noise_cells is None:
            noise_cells = self._noise_cells(self.config)
            self._lookup_noise_cells = noise_cells
        if self._lookup is None:
            self._lookup = FastOccupancyLookup(
                self.grid, freespace_noise_cells=noise_cells
            )
        return self._lookup

    @property
    def map_corners(self) -> tuple[CornerFeature, ...]:
        if self._map_corners is None:
            self._map_corners = extract_map_corners(
                self.grid,
                min_corner_separation=self.config.min_corner_separation,
                max_corners=None,
            )
        return self._map_corners

    def localize(
        self,
        scan: LidarScan,
        *,
        lidar_config: LidarConfig | None = None,
    ) -> LocalizationResult:
        cfg = self.config
        lidar_cfg = lidar_config or LidarConfig()

        scan_corners = extract_scan_corners(
            scan,
            range_min=lidar_cfg.range_min,
            range_max=lidar_cfg.range_max,
            corner_angle_min=cfg.corner_angle_min,
            min_corner_separation=cfg.min_corner_separation,
            max_corners=cfg.max_scan_corners,
        )
        map_corners = self.map_corners

        if not scan_corners:
            return LocalizationResult(
                pose=None,
                score=0.0,
                scan_corners=scan_corners,
                map_corners=map_corners,
                hypotheses_tested=0,
            )

        points = scan_to_points(
            scan,
            range_min=lidar_cfg.range_min,
            range_max=lidar_cfg.range_max,
        )
        search_stride = max(1, cfg.score_ray_stride)
        final_stride = max(1, cfg.final_score_ray_stride)
        search_xy = np.asarray(points[::search_stride], dtype=np.float64)
        final_xy = (
            search_xy
            if final_stride == search_stride
            else np.asarray(points[::final_stride], dtype=np.float64)
        )

        hit_cells = max(1, math.ceil(cfg.hit_tolerance / self.grid.resolution))
        self._lookup_noise_cells = self._noise_cells(cfg)

        # Rank and refine without per-ray freespace rejection so near-correct
        # corner poses are not zeroed while a wrong global alignment still scores well.
        search_scorer = PoseScorer(
            self.lookup,
            search_xy,
            scan_corners,
            map_corners,
            hit_radius_cells=hit_cells,
            freespace_consistency=False,
            reject_robot_outside_free=False,
        )
        final_scorer = (
            PoseScorer(
                self.lookup,
                search_xy,
                scan_corners,
                map_corners,
                hit_radius_cells=hit_cells,
                freespace_consistency=cfg.freespace_consistency,
                reject_robot_outside_free=cfg.reject_robot_outside_free,
            )
            if final_xy is search_xy
            else PoseScorer(
                self.lookup,
                final_xy,
                scan_corners,
                map_corners,
                hit_radius_cells=hit_cells,
                freespace_consistency=cfg.freespace_consistency,
                reject_robot_outside_free=cfg.reject_robot_outside_free,
            )
        )

        corner_w = cfg.search_corner_weight
        min_ep_for_corners = cfg.min_endpoint_for_corner_rank

        def rank_pose(pose: Pose2D) -> float:
            endpoint = search_scorer.score_fast(pose)
            if endpoint < min_ep_for_corners:
                return endpoint
            if corner_w > 0.0 and scan_corners:
                return endpoint + corner_w * search_scorer.score_corners(pose)
            return endpoint

        greedy = greedy_localize(
            scan_corners,
            map_corners,
            rank_pose,
            min_score=cfg.min_match_score,
            early_exit_score=cfg.early_exit_score,
            try_heading_flip=cfg.try_heading_flip,
            max_scan_corners_for_pairs=cfg.max_scan_corners_for_pairs,
            top_k=cfg.refine_top_k,
            grid_resolution=self.grid.resolution,
        )

        hypotheses_tested = greedy.hypotheses_tried
        stopped_early = greedy.stopped_early
        candidates = list(greedy.top_candidates)
        if not candidates and greedy.pose is not None:
            candidates = [(greedy.score, greedy.pose)]

        if cfg.use_grid_search and (not candidates or candidates[0][0] < cfg.min_match_score):
            for pose in generate_grid_hypotheses(
                self.grid,
                xy_step=cfg.grid_xy_step,
                theta_step=cfg.grid_theta_step,
                max_poses=cfg.max_grid_hypotheses,
            ):
                hypotheses_tested += 1
                score = search_scorer.score_fast(pose)
                candidates.append((score, pose))
                candidates.sort(key=lambda item: item[0], reverse=True)
                candidates = candidates[: cfg.refine_top_k]
                if score >= cfg.min_match_score:
                    stopped_early = True
                    break

        refined_candidates: list[tuple[float, float, float, Pose2D]] = []
        best_pose: Pose2D | None = None
        best_score = 0.0
        best_search_ep = 0.0

        for _fast_score, pose in candidates[: cfg.refine_top_k]:
            if cfg.refine_poses:
                if cfg.refine_multiscale:
                    pose, _ = refine_pose_multiscale(
                        search_scorer,
                        pose,
                        translation_span=cfg.translation_span,
                        rotation_span=cfg.rotation_span,
                    )
                else:
                    pose, _ = refine_pose(
                        search_scorer,
                        pose,
                        translation_span=cfg.translation_span,
                        rotation_span=cfg.rotation_span,
                    )
            search_ep = search_scorer.score_fast(pose)
            if search_ep <= 0.0:
                continue
            corners = search_scorer.score_corners(pose)
            score = search_ep + search_scorer.corner_weight * corners
            corner_cost = search_scorer.corner_assignment_cost(pose)
            refined_candidates.append((score, corner_cost, search_ep, pose))

        if refined_candidates:
            best_score, _, best_search_ep, best_pose = _pick_best_candidate(
                refined_candidates, min_match_score=cfg.min_match_score
            )

        needs_grid = cfg.grid_search_on_failure and (
            best_pose is None
            or best_score < cfg.min_match_score
            or best_search_ep < cfg.grid_search_endpoint_threshold
        )
        if needs_grid and (cfg.use_grid_search or cfg.grid_search_on_failure):
            grid_heap: list[tuple[float, int, Pose2D]] = []
            tie = 0
            keep_grid = max(3, cfg.refine_top_k // 2)
            for pose in generate_grid_hypotheses(
                self.grid,
                xy_step=cfg.grid_xy_step,
                theta_step=cfg.grid_theta_step,
                max_poses=cfg.max_grid_hypotheses,
            ):
                hypotheses_tested += 1
                fast = rank_pose(pose)
                tie += 1
                entry = (fast, tie, pose)
                if len(grid_heap) < keep_grid:
                    heapq.heappush(grid_heap, entry)
                elif fast > grid_heap[0][0]:
                    heapq.heapreplace(grid_heap, entry)

            for _fast, _, pose in sorted(grid_heap, key=lambda item: item[0], reverse=True):
                if cfg.refine_poses:
                    if cfg.refine_multiscale:
                        pose, _ = refine_pose_multiscale(
                            search_scorer,
                            pose,
                            translation_span=cfg.translation_span,
                            rotation_span=cfg.rotation_span,
                        )
                    else:
                        pose, _ = refine_pose(
                            search_scorer,
                            pose,
                            translation_span=cfg.translation_span,
                            rotation_span=cfg.rotation_span,
                        )
                search_ep = search_scorer.score_fast(pose)
                if search_ep <= 0.0:
                    continue
                corners = search_scorer.score_corners(pose)
                score = search_ep + search_scorer.corner_weight * corners
                corner_cost = search_scorer.corner_assignment_cost(pose)
                refined_candidates.append((score, corner_cost, search_ep, pose))

            if refined_candidates:
                best_score, _, best_search_ep, best_pose = _pick_best_candidate(
                    refined_candidates, min_match_score=cfg.min_match_score
                )

        if best_pose is None or best_score < cfg.min_match_score:
            return LocalizationResult(
                pose=None,
                score=max(best_score, 0.0),
                scan_corners=scan_corners,
                map_corners=map_corners,
                hypotheses_tested=hypotheses_tested,
                stopped_early=stopped_early,
            )

        return LocalizationResult(
            pose=best_pose,
            score=best_score,
            scan_corners=scan_corners,
            map_corners=map_corners,
            hypotheses_tested=hypotheses_tested,
            stopped_early=stopped_early,
        )
