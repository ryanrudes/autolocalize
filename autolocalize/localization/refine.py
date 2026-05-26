from __future__ import annotations

from autolocalize.geometry.pose import Pose2D
from autolocalize.localization.fast_grid import PoseScorer


def refine_pose(
    scorer: PoseScorer,
    pose: Pose2D,
    *,
    translation_span: float = 0.2,
    rotation_span: float = 0.35,
) -> tuple[Pose2D, float]:
    """Local grid search around a pose to maximize endpoint score."""
    return _search_grid(
        scorer,
        pose,
        translation_step=0.08,
        translation_span=translation_span,
        rotation_step=0.08,
        rotation_span=rotation_span,
    )


def refine_pose_quick(
    scorer: PoseScorer,
    pose: Pose2D,
    *,
    translation_span: float = 0.16,
    rotation_span: float = 0.28,
) -> tuple[Pose2D, float]:
    """Coarser, smaller local search for adaptive tier 1/2 (fewer evaluations)."""
    return _search_grid(
        scorer,
        pose,
        translation_step=0.1,
        translation_span=translation_span,
        rotation_step=0.1,
        rotation_span=rotation_span,
    )


def refine_pose_multiscale(
    scorer: PoseScorer,
    pose: Pose2D,
    *,
    translation_span: float = 0.2,
    rotation_span: float = 0.35,
) -> tuple[Pose2D, float]:
    """Coarse-to-fine grid search for larger initial pose error."""
    best_pose, best_score = pose, scorer.score_fast(pose)
    scales = (
        (max(translation_span, 0.35), 0.12, max(rotation_span, 0.45), 0.12),
        (0.15, 0.06, 0.25, 0.06),
        (0.06, 0.03, 0.12, 0.04),
    )
    for t_span, t_step, r_span, r_step in scales:
        candidate, score = _search_grid(
            scorer,
            best_pose,
            translation_step=t_step,
            translation_span=t_span,
            rotation_step=r_step,
            rotation_span=r_span,
        )
        if score > best_score:
            best_pose, best_score = candidate, score
    return best_pose, best_score


def _search_grid(
    scorer: PoseScorer,
    pose: Pose2D,
    *,
    translation_step: float,
    translation_span: float,
    rotation_step: float,
    rotation_span: float,
) -> tuple[Pose2D, float]:
    best_pose = pose
    best_score = scorer.score_fast(pose)

    for dx in _frange(-translation_span, translation_span, translation_step):
        for dy in _frange(-translation_span, translation_span, translation_step):
            for dtheta in _frange(-rotation_span, rotation_span, rotation_step):
                if dx == 0.0 and dy == 0.0 and dtheta == 0.0:
                    continue
                candidate = Pose2D(
                    x=pose.x + dx,
                    y=pose.y + dy,
                    theta=Pose2D.normalize_angle(pose.theta + dtheta),
                )
                score = scorer.score_fast(candidate)
                if score > best_score:
                    best_score = score
                    best_pose = candidate

    return best_pose, best_score


def _frange(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        return [start]
    values: list[float] = []
    v = start
    while v <= stop + step * 0.5:
        values.append(v)
        v += step
    return values
