from __future__ import annotations

import heapq
import itertools
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

from autolocalize.features.corners import CornerFeature
from autolocalize.geometry.pose import Pose2D
from autolocalize.geometry.transform import pose_from_correspondences
from autolocalize.localization.hypotheses import _poses_from_corner_pair


@dataclass(frozen=True, slots=True)
class GreedySearchResult:
    pose: Pose2D | None
    score: float
    hypotheses_tried: int
    stopped_early: bool
    top_candidates: tuple[tuple[float, Pose2D], ...] = ()


def greedy_localize(
    scan_corners: tuple[CornerFeature, ...],
    map_corners: tuple[CornerFeature, ...],
    score_pose: Callable[[Pose2D], float],
    *,
    min_score: float,
    early_exit_score: float | None = None,
    try_heading_flip: bool = True,
    max_scan_corners_for_pairs: int = 12,
    max_scan_corners_for_singles: int | None = None,
    top_k: int = 12,
    distance_tolerance: float | None = None,
    grid_resolution: float = 0.05,
) -> GreedySearchResult:
    """
    Try high-value feature correspondences first; keep the top-K scoring poses.

    Pair matches run before single-corner fallbacks. Optional early exit when a
    pose reaches ``early_exit_score``.
    """
    if not scan_corners or not map_corners:
        return GreedySearchResult(None, 0.0, 0, False, ())

    if distance_tolerance is None:
        distance_tolerance = max(0.12, grid_resolution * 4.0)

    stop_score = early_exit_score if early_exit_score is not None else float("inf")
    ordered_scan = sorted(scan_corners, key=lambda c: c.sharpness, reverse=True)
    tried = 0
    best_pose: Pose2D | None = None
    best_score = -1.0
    top_heap: list[tuple[float, int, Pose2D]] = []
    tie_counter = 0
    keep_k = max(1, top_k)

    def remember(score: float, pose: Pose2D) -> None:
        nonlocal best_pose, best_score, tie_counter
        tie_counter += 1
        entry = (score, tie_counter, pose)
        if len(top_heap) < keep_k:
            heapq.heappush(top_heap, entry)
        elif score > top_heap[0][0]:
            heapq.heapreplace(top_heap, entry)
        if score > best_score:
            best_score = score
            best_pose = pose

    def evaluate(pose: Pose2D, *, allow_early_exit: bool) -> float:
        nonlocal tried
        tried += 1
        score = score_pose(pose)
        remember(score, pose)
        if allow_early_exit and score >= stop_score:
            raise _EarlyExit(pose)
        return score

    try:
        if len(ordered_scan) >= 2:
            pair_scan = ordered_scan[:max_scan_corners_for_pairs]
            length_index = _build_length_index(map_corners, distance_tolerance)

            for s1, s2 in itertools.combinations(pair_scan, 2):
                d_scan = s1.distance_to(s2)
                if d_scan < 0.15:
                    continue
                for m1, m2 in _lookup_pairs(
                    d_scan, length_index, distance_tolerance
                ):
                    for ma, mb in ((m1, m2), (m2, m1)):
                        pose = pose_from_correspondences(
                            (s1.x, s1.y),
                            (s2.x, s2.y),
                            (ma.x, ma.y),
                            (mb.x, mb.y),
                        )
                        evaluate(pose, allow_early_exit=True)

        single_scan = ordered_scan
        if max_scan_corners_for_singles is not None:
            single_scan = ordered_scan[: max(1, max_scan_corners_for_singles)]

        for sc in single_scan:
            for mc in map_corners:
                for pose in _poses_from_corner_pair(sc, mc, try_heading_flip):
                    evaluate(pose, allow_early_exit=False)

    except _EarlyExit as done:
        top = _heap_to_sorted(top_heap)
        return GreedySearchResult(done.pose, best_score, tried, True, top)

    top = _heap_to_sorted(top_heap)
    return GreedySearchResult(best_pose, max(best_score, 0.0), tried, False, top)


class _EarlyExit(Exception):
    __slots__ = ("pose",)

    def __init__(self, pose: Pose2D) -> None:
        self.pose = pose


def _heap_to_sorted(heap: list[tuple[float, int, Pose2D]]) -> tuple[tuple[float, Pose2D], ...]:
    return tuple(
        (score, pose)
        for score, _, pose in sorted(heap, key=lambda item: item[0], reverse=True)
    )


def _build_length_index(
    map_corners: tuple[CornerFeature, ...],
    tolerance: float,
) -> dict[int, list[tuple[CornerFeature, CornerFeature]]]:
    index: dict[int, list[tuple[CornerFeature, CornerFeature]]] = defaultdict(list)
    bucket = max(tolerance, 0.05)
    for m1, m2 in itertools.combinations(map_corners, 2):
        d = m1.distance_to(m2)
        if d < 0.15:
            continue
        key = int(round(d / bucket))
        index[key].append((m1, m2))
    return index


def _lookup_pairs(
    d_scan: float,
    index: dict[int, list[tuple[CornerFeature, CornerFeature]]],
    tolerance: float,
) -> list[tuple[CornerFeature, CornerFeature]]:
    bucket = max(tolerance, 0.05)
    key = int(round(d_scan / bucket))
    candidates: list[tuple[CornerFeature, CornerFeature]] = []
    for k in (key - 1, key, key + 1):
        for m1, m2 in index.get(k, []):
            if abs(m1.distance_to(m2) - d_scan) <= tolerance:
                candidates.append((m1, m2))
    return candidates
