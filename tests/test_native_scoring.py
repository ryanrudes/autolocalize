"""Tests that native C++ scoring matches the Python reference implementation."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from autolocalize import LidarConfig, LidarSimulator, Pose2D, load_map
from autolocalize.features.map_features import extract_map_corners
from autolocalize.features.scan import extract_scan_corners, scan_to_points
from autolocalize.localization import fast_grid
from autolocalize.localization.fast_grid import FastOccupancyLookup, PoseScorer

pytest.importorskip("autolocalize._native")

MAP_PATH = Path(__file__).resolve().parents[1] / "maps" / "churchsidemaze1.yaml"


def _make_scorer(*, freespace: bool) -> tuple[PoseScorer, list[Pose2D]]:
    maze = load_map(MAP_PATH)
    cfg = LidarConfig(num_rays=360, range_min=0.05, range_max=4.0)
    true_pose = Pose2D(0.0, 0.0, 0.0)
    wrong_pose = Pose2D(1.5, -1.0, math.pi / 2)
    shifted_pose = Pose2D(0.12, -0.08, 0.25)
    scan = LidarSimulator(maze, cfg).scan(true_pose)
    lookup = FastOccupancyLookup(maze)
    points = scan_to_points(scan, range_min=cfg.range_min, range_max=cfg.range_max)
    map_corners = extract_map_corners(maze)
    scan_corners = extract_scan_corners(
        scan, range_min=cfg.range_min, range_max=cfg.range_max
    )
    scorer = PoseScorer(
        lookup,
        np.asarray(points[::6], dtype=np.float64),
        scan_corners,
        map_corners,
        hit_radius_cells=2,
        freespace_consistency=freespace,
        reject_robot_outside_free=freespace,
    )
    assert scorer.uses_native
    return scorer, [true_pose, wrong_pose, shifted_pose]


def _python_score_fast(scorer: PoseScorer, pose: Pose2D) -> float:
    native = scorer._native
    scorer._native = None
    try:
        return scorer._score_endpoints(pose)
    finally:
        scorer._native = native


def _python_score_corners(scorer: PoseScorer, pose: Pose2D) -> float:
    native = scorer._native
    scorer._native = None
    try:
        n_scan = scorer._scan_xy.shape[0]
        if n_scan == 0 or scorer._map_xy.shape[0] == 0:
            return 0.0
        wxwy = fast_grid.transform_points(pose, scorer._scan_xy)
        world_angles = np.remainder(
            pose.theta + scorer._scan_angles + math.pi, math.tau
        ) - math.pi
        diff = wxwy[:, None, :] - scorer._map_xy[None, :, :]
        dist_sq = diff[..., 0] ** 2 + diff[..., 1] ** 2
        if scorer.corner_match_requires_angle:
            angle_diff = np.abs(
                np.remainder(
                    world_angles[:, None] - scorer._map_angles[None, :] + math.pi,
                    math.tau,
                )
                - math.pi
            )
            matched = np.any(
                (dist_sq <= scorer._pos_tol_sq)
                & (angle_diff <= scorer.angle_tolerance),
                axis=1,
            )
        else:
            matched = np.any(dist_sq <= scorer._pos_tol_sq, axis=1)
        return float(np.count_nonzero(matched)) / n_scan
    finally:
        scorer._native = native


@pytest.mark.parametrize("freespace", [False, True])
def test_native_score_fast_matches_python(freespace: bool) -> None:
    scorer, poses = _make_scorer(freespace=freespace)
    for pose in poses:
        native = scorer.score_fast(pose)
        reference = _python_score_fast(scorer, pose)
        assert native == pytest.approx(reference, rel=0.0, abs=1e-15)


@pytest.mark.parametrize("freespace", [False, True])
def test_native_score_corners_matches_python(freespace: bool) -> None:
    scorer, poses = _make_scorer(freespace=freespace)
    for pose in poses:
        native = scorer.score_corners(pose)
        reference = _python_score_corners(scorer, pose)
        assert native == pytest.approx(reference, rel=0.0, abs=1e-15)


def test_native_module_is_available() -> None:
    from autolocalize.localization.fast_grid import PoseScorerNative

    assert PoseScorerNative is not None
