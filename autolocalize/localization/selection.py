from __future__ import annotations

from autolocalize.geometry.pose import Pose2D


def pick_best_candidate(
    candidates: list[tuple[float, float, float, Pose2D]],
    *,
    min_match_score: float,
    strong_endpoint: float = 0.92,
    endpoint_tie_band: float = 0.03,
) -> tuple[float, float, float, Pose2D]:
    """
    Choose a pose from refined candidates.

    Prefer high endpoint match scores; among near-ties in endpoint score use
    lowest corner assignment cost. Endpoint-led disambiguation avoids picking a
    mirrored alias that only wins on corner cost.
    """
    ranked = sorted(candidates, key=lambda item: (-item[0], item[1]))
    best_score, _, best_search_ep, best_pose = ranked[0]
    strong = [item for item in ranked if item[2] >= strong_endpoint]
    if len(strong) >= 2:
        max_ep = max(item[2] for item in strong)
        leaders = [item for item in strong if item[2] >= max_ep - endpoint_tie_band]
        leaders.sort(key=lambda item: (item[1], -item[0]))
        best_score, _, best_search_ep, best_pose = leaders[0]
    elif best_score < min_match_score and len(ranked) > 1:
        for score, cost, search_ep, pose in ranked[1:]:
            if score >= min_match_score:
                return score, cost, search_ep, pose
    return best_score, 0.0, best_search_ep, best_pose
