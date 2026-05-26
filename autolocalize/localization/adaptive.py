from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from autolocalize.features.corners import CornerFeature
from autolocalize.geometry.pose import Pose2D
from autolocalize.localization.fast_grid import PoseScorer
from autolocalize.localization.greedy import greedy_localize
from autolocalize.localization.hypotheses import generate_grid_hypotheses
from autolocalize.localization.config import InitialLocalizerConfig
from autolocalize.localization.selection import pick_best_candidate
from autolocalize.localization.refine import refine_pose, refine_pose_multiscale

if TYPE_CHECKING:
    from autolocalize.map.grid import OccupancyGrid


@dataclass(frozen=True, slots=True)
class AdaptiveOutcome:
  pose: Pose2D | None
  score: float
  hypotheses_tested: int
  stopped_early: bool
  effort_tier: int


def _candidate_tuple(
    scorer: PoseScorer,
    pose: Pose2D,
    *,
    refine: Callable[[Pose2D], Pose2D] | None = None,
) -> tuple[float, float, float, Pose2D] | None:
    if refine is not None:
        pose = refine(pose)
    search_ep = scorer.score_fast(pose)
    if search_ep <= 0.0:
        return None
    corners = scorer.score_corners(pose)
    score = search_ep + scorer.corner_weight * corners
    corner_cost = scorer.corner_assignment_cost(pose)
    return (score, corner_cost, search_ep, pose)


def _rank_raw(
    scorer: PoseScorer, candidates: list[tuple[float, Pose2D]]
) -> list[tuple[float, float, float, Pose2D]]:
    ranked: list[tuple[float, float, float, Pose2D]] = []
    for _, pose in candidates:
        row = _candidate_tuple(scorer, pose)
        if row is not None:
            ranked.append(row)
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return ranked


def _score_margin(ranked: list[tuple[float, float, float, Pose2D]]) -> float:
    if len(ranked) < 2:
        return float("inf")
    return ranked[0][0] - ranked[1][0]


def _count_strong(
    ranked: list[tuple[float, float, float, Pose2D]], *, threshold: float
) -> int:
    return sum(1 for row in ranked if row[2] >= threshold)


def _is_confident(
    ranked: list[tuple[float, float, float, Pose2D]],
    *,
    min_ep: float,
    min_margin: float,
    strong_ep: float,
) -> bool:
    if not ranked:
        return False
    best_ep = ranked[0][2]
    if best_ep < min_ep:
        return False
    if _score_margin(ranked) < min_margin:
        return False
    return _count_strong(ranked, threshold=strong_ep) <= 1


def _refine_single(
    scorer: PoseScorer,
    pose: Pose2D,
    cfg: InitialLocalizerConfig,
    *,
    multiscale: bool,
) -> Pose2D:
    if multiscale:
        pose, _ = refine_pose_multiscale(
            scorer,
            pose,
            translation_span=cfg.translation_span,
            rotation_span=cfg.rotation_span,
        )
    else:
        pose, _ = refine_pose(
            scorer,
            pose,
            translation_span=cfg.translation_span,
            rotation_span=cfg.rotation_span,
        )
    return pose


def _refine_candidates(
    scorer: PoseScorer,
    candidates: list[tuple[float, Pose2D]],
    cfg: InitialLocalizerConfig,
    *,
    limit: int,
    multiscale: bool,
) -> list[tuple[float, float, float, Pose2D]]:
    refined: list[tuple[float, float, float, Pose2D]] = []

    def do_refine(p: Pose2D) -> Pose2D:
        return _refine_single(scorer, p, cfg, multiscale=multiscale)

    for _, pose in candidates[:limit]:
        row = _candidate_tuple(scorer, pose, refine=do_refine)
        if row is not None:
            refined.append(row)
    return refined


def _run_grid_recovery(
    grid: OccupancyGrid,
    scorer: PoseScorer,
    rank_pose: Callable[[Pose2D], float],
    cfg: InitialLocalizerConfig,
    *,
    hypotheses_tested: int,
    refined_candidates: list[tuple[float, float, float, Pose2D]],
) -> tuple[list[tuple[float, float, float, Pose2D]], int]:
    grid_heap: list[tuple[float, int, Pose2D]] = []
    tie = 0
    keep_grid = max(3, cfg.adaptive_tier3_top_k // 2)

    for pose in generate_grid_hypotheses(
        grid,
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

    grid_candidates = [
        (score, pose) for score, _, pose in sorted(grid_heap, key=lambda e: e[0], reverse=True)
    ]
    refined_candidates.extend(
        _refine_candidates(
            scorer,
            grid_candidates,
            cfg,
            limit=cfg.adaptive_tier3_top_k,
            multiscale=True,
        )
    )
    return refined_candidates, hypotheses_tested


def localize_adaptive(
    grid: OccupancyGrid,
    scan_corners: tuple[CornerFeature, ...],
    map_corners: tuple[CornerFeature, ...],
    search_scorer: PoseScorer,
    cfg: InitialLocalizerConfig,
    rank_pose: Callable[[Pose2D], float],
) -> AdaptiveOutcome:
    """
    Tiered localization: cheap greedy + scoring first, deepen only when uncertain.

    Tier 0 — quick greedy, raw scores only
    Tier 1 — single-scale refine on top-1
    Tier 2 — refine top-K + disambiguation
    Tier 3 — full greedy + multiscale refine (+ grid if still weak)
    """
    hypotheses_tested = 0
    stopped_early = False

    greedy_quick = greedy_localize(
        scan_corners,
        map_corners,
        rank_pose,
        min_score=cfg.min_match_score,
        early_exit_score=cfg.adaptive_early_exit_score,
        try_heading_flip=cfg.try_heading_flip,
        max_scan_corners_for_pairs=cfg.adaptive_quick_pairs,
        max_scan_corners_for_singles=cfg.adaptive_quick_singles,
        top_k=cfg.adaptive_quick_top_k,
        grid_resolution=grid.resolution,
    )
    hypotheses_tested += greedy_quick.hypotheses_tried
    stopped_early = greedy_quick.stopped_early
    candidates = list(greedy_quick.top_candidates)
    if not candidates and greedy_quick.pose is not None:
        candidates = [(greedy_quick.score, greedy_quick.pose)]

    if not candidates:
        return AdaptiveOutcome(None, 0.0, hypotheses_tested, stopped_early, 0)

    raw = _rank_raw(search_scorer, candidates)

    # Tier 0: very confident without any refine
    if _is_confident(
        raw,
        min_ep=cfg.adaptive_tier0_min_ep,
        min_margin=cfg.adaptive_confident_margin,
        strong_ep=cfg.adaptive_strong_ep,
    ):
        best_score, _, _, best_pose = raw[0]
        if best_score >= cfg.min_match_score:
            return AdaptiveOutcome(
                best_pose, best_score, hypotheses_tested, stopped_early, 0
            )

    # Tier 1: one quick refine on the best raw candidate
    tier1 = _refine_candidates(
        search_scorer, candidates, cfg, limit=1, multiscale=False
    )
    if tier1 and _is_confident(
        tier1,
        min_ep=cfg.adaptive_tier1_min_ep,
        min_margin=cfg.adaptive_confident_margin,
        strong_ep=cfg.adaptive_strong_ep,
    ):
        best_score, _, best_search_ep, best_pose = tier1[0]
        if best_score >= cfg.min_match_score:
            return AdaptiveOutcome(
                best_pose, best_score, hypotheses_tested, stopped_early, 1
            )

    ambiguous = _count_strong(raw, threshold=cfg.adaptive_strong_ep) >= 2

    # Tier 2: refine a few candidates and pick with corner-cost tie-break
    if ambiguous or (tier1 and tier1[0][2] < cfg.adaptive_tier2_min_ep):
        tier2 = _refine_candidates(
            search_scorer,
            candidates,
            cfg,
            limit=cfg.adaptive_tier2_top_k,
            multiscale=False,
        )
        if tier2:
            best_score, _, best_search_ep, best_pose = pick_best_candidate(
                tier2, min_match_score=cfg.min_match_score
            )
            if (
                best_pose is not None
                and best_score >= cfg.min_match_score
                and best_search_ep >= cfg.adaptive_tier2_accept_ep
            ):
                return AdaptiveOutcome(
                    best_pose, best_score, hypotheses_tested, stopped_early, 2
                )

    # Tier 3: full search + multiscale (+ optional grid)
    greedy_full = greedy_localize(
        scan_corners,
        map_corners,
        rank_pose,
        min_score=cfg.min_match_score,
        early_exit_score=None,
        try_heading_flip=cfg.try_heading_flip,
        max_scan_corners_for_pairs=cfg.max_scan_corners_for_pairs,
        max_scan_corners_for_singles=None,
        top_k=cfg.adaptive_tier3_top_k,
        grid_resolution=grid.resolution,
    )
    hypotheses_tested += greedy_full.hypotheses_tried
    full_candidates = list(greedy_full.top_candidates)
    if not full_candidates and greedy_full.pose is not None:
        full_candidates = [(greedy_full.score, greedy_full.pose)]

    refined = _refine_candidates(
        search_scorer,
        full_candidates,
        cfg,
        limit=cfg.adaptive_tier3_top_k,
        multiscale=True,
    )

    best_pose: Pose2D | None = None
    best_score = 0.0
    best_search_ep = 0.0
    if refined:
        best_score, _, best_search_ep, best_pose = pick_best_candidate(
            refined, min_match_score=cfg.min_match_score
        )

    needs_grid = cfg.grid_search_on_failure and (
        best_pose is None
        or best_score < cfg.min_match_score
        or best_search_ep < cfg.grid_search_endpoint_threshold
    )
    if needs_grid:
        refined, hypotheses_tested = _run_grid_recovery(
            grid,
            search_scorer,
            rank_pose,
            cfg,
            hypotheses_tested=hypotheses_tested,
            refined_candidates=refined,
        )
        if refined:
            best_score, _, best_search_ep, best_pose = pick_best_candidate(
                refined, min_match_score=cfg.min_match_score
            )

    if best_pose is None or best_score < cfg.min_match_score:
        return AdaptiveOutcome(
            None, max(best_score, 0.0), hypotheses_tested, stopped_early, 3
        )

    return AdaptiveOutcome(
        best_pose, best_score, hypotheses_tested, stopped_early, 3
    )
