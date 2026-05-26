from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


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
    refine_icp: bool = False
    icp_max_iterations: int = 20
    icp_max_association_dist: float = 0.25
    icp_convergence_translation: float = 1e-4
    icp_convergence_rotation: float = 1e-4
    icp_huber_delta: float = 0.05
    icp_min_points: int = 20
    icp_ray_stride: int = 1
    icp_bucket_size: float = 0.5
    effort: Literal["standard", "fast", "adaptive"] = "standard"
    # Adaptive effort (refine-before-accept; no unrefined early exit)
    adaptive_quick_stride: int = 8
    adaptive_quick_pairs: int = 6
    adaptive_quick_singles: int = 6
    adaptive_quick_map_singles: int = 28
    adaptive_quick_top_k: int = 8
    adaptive_tier1_refine_k: int = 3
    adaptive_tier2_refine_k: int = 5
    adaptive_tier3_top_k: int = 8
    adaptive_quick_win_ep: float = 0.97
    adaptive_quick_win_endpoint_gap: float = 0.035
    adaptive_confident_min_ep: float = 0.93
    adaptive_confident_margin: float = 0.04
    adaptive_confident_endpoint_margin: float = 0.025
    adaptive_strong_ep: float = 0.88
    adaptive_tier2_accept_ep: float = 0.91
    adaptive_tier3_trigger_ep: float = 0.86
    adaptive_position_alias_min_m: float = 0.45
    adaptive_corner_cost_margin_min: float = 0.03
    adaptive_early_min_corners: float = 0.52
    adaptive_early_max_corner_cost: float = 0.40
    adaptive_early_min_ep_with_weak_corners: float = 0.975
    adaptive_early_min_corners_with_weak_ep: float = 0.35
    # Legacy names kept for compatibility (unused by new adaptive path)
    adaptive_early_exit_score: float | None = None
    adaptive_tier0_min_ep: float = 0.97
    adaptive_tier1_min_ep: float = 0.95
    adaptive_tier2_min_ep: float = 0.88
    adaptive_tier2_top_k: int = 3


def config_for_effort(effort: Literal["standard", "fast", "adaptive"]) -> InitialLocalizerConfig:
    """Presets for standard, fast, or adaptive localization effort."""
    if effort == "fast":
        return InitialLocalizerConfig(
            effort="fast",
            use_grid_search=False,
            score_ray_stride=8,
            max_scan_corners_for_pairs=4,
            refine_poses=False,
            grid_search_on_failure=False,
        )
    if effort == "adaptive":
        return InitialLocalizerConfig(
            effort="adaptive",
            use_grid_search=False,
            score_ray_stride=8,
            grid_search_on_failure=True,
        )
    return InitialLocalizerConfig(effort="standard", use_grid_search=False)
