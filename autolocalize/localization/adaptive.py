from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from autolocalize.features.corners import CornerFeature
from autolocalize.geometry.pose import Pose2D
from autolocalize.localization.config import InitialLocalizerConfig
from autolocalize.localization.fast_grid import PoseScorer
from autolocalize.localization.greedy import greedy_localize
from autolocalize.localization.hypotheses import generate_grid_hypotheses
from autolocalize.localization.refine import (
    refine_pose,
    refine_pose_multiscale,
    refine_pose_quick,
)
from autolocalize.localization.selection import pick_best_candidate

if TYPE_CHECKING:
    from autolocalize.map.grid import OccupancyGrid

CandidateRow = tuple[float, float, float, Pose2D]


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
) -> CandidateRow | None:
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
) -> list[CandidateRow]:
    ranked: list[CandidateRow] = []
    for _, pose in candidates:
        row = _candidate_tuple(scorer, pose)
        if row is not None:
            ranked.append(row)
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return ranked


def _score_margin(rows: list[CandidateRow]) -> float:
    if len(rows) < 2:
        return float("inf")
    return rows[0][0] - rows[1][0]


def _endpoint_margin(rows: list[CandidateRow]) -> float:
    if len(rows) < 2:
        return float("inf")
    return rows[0][2] - rows[1][2]


def _pose_distance(a: Pose2D, b: Pose2D) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _position_spread(poses: list[Pose2D]) -> float:
    if len(poses) < 2:
        return 0.0
    max_d = 0.0
    for i, p1 in enumerate(poses):
        for p2 in poses[i + 1 :]:
            max_d = max(max_d, _pose_distance(p1, p2))
    return max_d


def _strong_rows(rows: list[CandidateRow], *, threshold: float) -> list[CandidateRow]:
    return [row for row in rows if row[2] >= threshold]


def _has_position_alias(
    rows: list[CandidateRow],
    *,
    strong_ep: float,
    min_separation_m: float,
) -> bool:
    """True when multiple high-scoring hypotheses disagree by translation."""
    strong = _strong_rows(rows, threshold=strong_ep)
    if len(strong) < 2:
        return False
    return _position_spread([row[3] for row in strong]) >= min_separation_m


def _corner_cost_margin(rows: list[CandidateRow]) -> float:
    if len(rows) < 2:
        return float("inf")
    ordered = sorted(rows, key=lambda item: item[1])
    return ordered[1][1] - ordered[0][1]


def _has_clear_winner(
    rows: list[CandidateRow],
    cfg: InitialLocalizerConfig,
    *,
    min_ep: float,
    min_endpoint_gap: float,
) -> bool:
    """Single dominant hypothesis after refine (endpoint-led)."""
    if not rows:
        return False
    if rows[0][2] < min_ep:
        return False
    if _has_position_alias(
        rows,
        strong_ep=cfg.adaptive_strong_ep,
        min_separation_m=cfg.adaptive_position_alias_min_m,
    ):
        return False
    if len(rows) >= 2 and _endpoint_margin(rows) < min_endpoint_gap:
        return False
    return True


def _is_confident_after_refine(
    rows: list[CandidateRow],
    cfg: InitialLocalizerConfig,
) -> bool:
    """Strong match with geometric agreement among refined poses."""
    if not _has_clear_winner(
        rows,
        cfg,
        min_ep=cfg.adaptive_confident_min_ep,
        min_endpoint_gap=cfg.adaptive_confident_endpoint_margin,
    ):
        return False
    if len(rows) >= 2 and _score_margin(rows) < cfg.adaptive_confident_margin:
        return False
    strong = _strong_rows(rows, threshold=cfg.adaptive_strong_ep)
    if len(strong) >= 2 and _corner_cost_margin(strong) < cfg.adaptive_corner_cost_margin_min:
        return False
    return True


def _tier2_acceptable(rows: list[CandidateRow], cfg: InitialLocalizerConfig) -> bool:
    if not rows:
        return False
    best_score, _, best_ep, _ = rows[0]
    if best_score < cfg.min_match_score:
        return False
    return _has_clear_winner(
        rows,
        cfg,
        min_ep=cfg.adaptive_tier2_accept_ep,
        min_endpoint_gap=cfg.adaptive_confident_endpoint_margin,
    )


def _early_exit_corners_ok(
    scorer: PoseScorer,
    pose: Pose2D,
    corner_cost: float,
    endpoint: float,
    cfg: InitialLocalizerConfig,
) -> bool:
    """Block tier 1/2 early exit on corridor aliases (strong ep, weak corners)."""
    corners = scorer.score_corners(pose)
    if corners >= cfg.adaptive_early_min_corners:
        return True
    if (
        endpoint >= cfg.adaptive_early_min_ep_with_weak_corners
        and corners >= cfg.adaptive_early_min_corners_with_weak_ep
    ):
        return True
    if corner_cost > cfg.adaptive_early_max_corner_cost:
        return False
    return False


def _tier2_resolved_by_selection(
    rows: list[CandidateRow], cfg: InitialLocalizerConfig
) -> bool:
    """
    Accept tier 2 when endpoint-led pick is strong and top-scoring poses agree.

    Accuracy comes from pick_best_candidate's endpoint tie-band; we only escalate
    when multiple poses in that band still disagree by translation.
    """
    if not rows:
        return False
    _, _, best_ep, _ = pick_best_candidate(
        rows,
        min_match_score=cfg.min_match_score,
        strong_endpoint=cfg.adaptive_strong_ep,
        endpoint_tie_band=cfg.adaptive_confident_endpoint_margin,
    )
    if best_ep < cfg.adaptive_tier2_accept_ep:
        return False
    max_ep = max(row[2] for row in rows)
    if best_ep < max_ep - cfg.adaptive_confident_endpoint_margin:
        return False
    leaders = [
        row
        for row in rows
        if row[2] >= max_ep - cfg.adaptive_confident_endpoint_margin
    ]
    if len(leaders) >= 2 and _position_spread([row[3] for row in leaders]) >= (
        cfg.adaptive_position_alias_min_m
    ):
        return False
    return True


def _refine_single(
    scorer: PoseScorer,
    pose: Pose2D,
    cfg: InitialLocalizerConfig,
    *,
    multiscale: bool,
    quick: bool,
) -> Pose2D:
    if multiscale:
        pose, _ = refine_pose_multiscale(
            scorer,
            pose,
            translation_span=cfg.translation_span,
            rotation_span=cfg.rotation_span,
        )
    elif quick:
        pose, _ = refine_pose_quick(scorer, pose)
    else:
        pose, _ = refine_pose(
            scorer,
            pose,
            translation_span=cfg.translation_span,
            rotation_span=cfg.rotation_span,
        )
    return pose


def _refine_ranked(
    scorer: PoseScorer,
    ranked: list[CandidateRow],
    cfg: InitialLocalizerConfig,
    *,
    limit: int,
    multiscale: bool,
    quick: bool = False,
) -> list[CandidateRow]:
    refined: list[CandidateRow] = []

    def do_refine(p: Pose2D) -> Pose2D:
        return _refine_single(scorer, p, cfg, multiscale=multiscale, quick=quick)

    for _, _, _, pose in ranked[:limit]:
        row = _candidate_tuple(scorer, pose, refine=do_refine)
        if row is not None:
            refined.append(row)
    refined.sort(key=lambda item: (-item[0], item[1]))
    return refined


def _tier1_exit_ok(rows: list[CandidateRow], cfg: InitialLocalizerConfig) -> bool:
    return _has_clear_winner(
        rows,
        cfg,
        min_ep=cfg.adaptive_quick_win_ep,
        min_endpoint_gap=cfg.adaptive_quick_win_endpoint_gap,
    ) or _is_confident_after_refine(rows, cfg)


def _raw_needs_disambiguation(
    raw: list[CandidateRow], cfg: InitialLocalizerConfig
) -> bool:
    """True when the quick greedy heap still has competing hypotheses."""
    top = raw[: cfg.adaptive_tier2_refine_k]
    if not top:
        return False
    if _has_position_alias(
        top,
        strong_ep=cfg.adaptive_strong_ep,
        min_separation_m=cfg.adaptive_position_alias_min_m,
    ):
        return True
    if len(top) >= 2 and _endpoint_margin(top) < cfg.adaptive_confident_endpoint_margin:
        return True
    close = [row for row in top if row[2] >= cfg.adaptive_tier3_trigger_ep]
    return len(close) >= 2 and _position_spread([row[3] for row in close]) >= (
        cfg.adaptive_position_alias_min_m * 0.5
    )


def _refine_tier1_cascade(
    scorer: PoseScorer,
    raw: list[CandidateRow],
    cfg: InitialLocalizerConfig,
) -> list[CandidateRow]:
    """Refine top-1 first; expand only when quick exit is not safe."""
    if not raw:
        return []

    limit = cfg.adaptive_tier1_refine_k
    refined = _refine_ranked(
        scorer, raw[:1], cfg, limit=1, multiscale=False, quick=True
    )
    if not refined:
        return []

    if _tier1_exit_ok(refined, cfg) and not _raw_needs_disambiguation(raw, cfg):
        return refined

    if len(raw) < 2 or limit <= 1:
        return refined

    if not _raw_needs_disambiguation(raw, cfg) and refined[0][2] >= cfg.adaptive_tier2_accept_ep:
        return refined

    return _refine_ranked(
        scorer, raw, cfg, limit=limit, multiscale=False, quick=True
    )


def _run_grid_recovery(
    grid: OccupancyGrid,
    scorer: PoseScorer,
    rank_pose: Callable[[Pose2D], float],
    cfg: InitialLocalizerConfig,
    *,
    hypotheses_tested: int,
    refined_candidates: list[CandidateRow],
) -> tuple[list[CandidateRow], int]:
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

    grid_poses = [
        pose for _, _, pose in sorted(grid_heap, key=lambda item: item[0], reverse=True)
    ]
    grid_ranked = _rank_raw(scorer, [(0.0, p) for p in grid_poses])
    refined_candidates.extend(
        _refine_ranked(
            scorer,
            grid_ranked,
            cfg,
            limit=cfg.adaptive_tier3_top_k,
            multiscale=True,
        )
    )
    refined_candidates.sort(key=lambda item: (-item[0], item[1]))
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
    Tiered localization with refine-before-accept and geometric ambiguity checks.

    Tier 1 — quick greedy (limited corners), refine top-ranked raw hypotheses
    Tier 2 — refine more ranked candidates + corner-cost disambiguation
    Tier 3 — full greedy, multiscale refine, optional grid recovery
    """
    hypotheses_tested = 0
    stopped_early = False

    greedy_quick = greedy_localize(
        scan_corners,
        map_corners,
        rank_pose,
        min_score=cfg.min_match_score,
        early_exit_score=None,
        try_heading_flip=cfg.try_heading_flip,
        max_scan_corners_for_pairs=cfg.adaptive_quick_pairs,
        max_scan_corners_for_singles=cfg.adaptive_quick_singles,
        max_map_corners_for_singles=cfg.adaptive_quick_map_singles,
        top_k=cfg.adaptive_quick_top_k,
        grid_resolution=grid.resolution,
    )
    hypotheses_tested += greedy_quick.hypotheses_tried
    stopped_early = greedy_quick.stopped_early
    heap_candidates = list(greedy_quick.top_candidates)
    if not heap_candidates and greedy_quick.pose is not None:
        heap_candidates = [(greedy_quick.score, greedy_quick.pose)]

    if not heap_candidates:
        return AdaptiveOutcome(None, 0.0, hypotheses_tested, stopped_early, 3)

    raw = _rank_raw(search_scorer, heap_candidates)
    if not raw:
        return AdaptiveOutcome(None, 0.0, hypotheses_tested, stopped_early, 3)

    # Tier 1 — cascade refine ranked hypotheses (never accept unrefined poses)
    tier1 = _refine_tier1_cascade(search_scorer, raw, cfg)
    raw_alias = _has_position_alias(
        raw[: cfg.adaptive_tier2_refine_k],
        strong_ep=cfg.adaptive_strong_ep,
        min_separation_m=cfg.adaptive_position_alias_min_m,
    )
    raw_ambiguous = _raw_needs_disambiguation(raw, cfg)
    if tier1 and _tier1_exit_ok(tier1, cfg) and not raw_alias and not raw_ambiguous:
        best_score, best_cost, best_ep, best_pose = pick_best_candidate(
            tier1,
            min_match_score=cfg.min_match_score,
            strong_endpoint=cfg.adaptive_strong_ep,
        )
        if _early_exit_corners_ok(
            search_scorer, best_pose, best_cost, best_ep, cfg
        ):
            return AdaptiveOutcome(
                best_pose, best_score, hypotheses_tested, stopped_early, 1
            )

    # Tier 2 — disambiguate when quick search is ambiguous or weak
    needs_tier2 = (
        raw_alias
        or raw_ambiguous
        or (tier1 and tier1[0][2] < cfg.adaptive_tier2_accept_ep)
    )

    tier2: list[CandidateRow] = []
    if needs_tier2:
        tier2 = _refine_ranked(
            search_scorer,
            raw,
            cfg,
            limit=cfg.adaptive_tier2_refine_k,
            multiscale=False,
            quick=True,
        )
        if tier2 and (
            _tier2_acceptable(tier2, cfg)
            or _tier2_resolved_by_selection(tier2, cfg)
        ):
            best_score, best_cost, best_ep, best_pose = pick_best_candidate(
                tier2,
                min_match_score=cfg.min_match_score,
                strong_endpoint=cfg.adaptive_strong_ep,
            )
            if _early_exit_corners_ok(
                search_scorer, best_pose, best_cost, best_ep, cfg
            ):
                return AdaptiveOutcome(
                    best_pose, best_score, hypotheses_tested, stopped_early, 2
                )

    # Tier 3 — full feature search + multiscale (+ grid when still weak)
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
    full_heap = list(greedy_full.top_candidates)
    if not full_heap and greedy_full.pose is not None:
        full_heap = [(greedy_full.score, greedy_full.pose)]

    full_raw = _rank_raw(search_scorer, full_heap)
    refined = _refine_ranked(
        search_scorer,
        full_raw,
        cfg,
        limit=cfg.adaptive_tier3_top_k,
        multiscale=True,
    )

    best_pose: Pose2D | None = None
    best_score = 0.0
    best_search_ep = 0.0
    if refined:
        best_score, _, best_search_ep, best_pose = pick_best_candidate(
            refined,
            min_match_score=cfg.min_match_score,
            strong_endpoint=cfg.adaptive_strong_ep,
        )

    needs_grid = cfg.grid_search_on_failure and (
        best_pose is None
        or best_score < cfg.min_match_score
        or best_search_ep < cfg.grid_search_endpoint_threshold
        or _has_position_alias(
            refined,
            strong_ep=cfg.adaptive_strong_ep,
            min_separation_m=cfg.adaptive_position_alias_min_m * 0.75,
        )
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
                refined,
                min_match_score=cfg.min_match_score,
                strong_endpoint=cfg.adaptive_strong_ep,
            )

    if best_pose is None or best_score < cfg.min_match_score:
        return AdaptiveOutcome(
            None, max(best_score, 0.0), hypotheses_tested, stopped_early, 3
        )

    return AdaptiveOutcome(
        best_pose, best_score, hypotheses_tested, stopped_early, 3
    )
