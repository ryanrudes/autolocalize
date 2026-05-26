import math

import numpy as np
import pytest
from pathlib import Path

from autolocalize import LidarConfig, LidarSimulator, Pose2D, load_map
from autolocalize.features.scan import scan_to_points
from autolocalize.localization.fast_grid import FastOccupancyLookup, PoseScorer
from autolocalize.localization.initial import InitialLocalizer, InitialLocalizerConfig

MAP_PATH = Path(__file__).resolve().parents[1] / "maps" / "churchsidemaze1.yaml"


def test_wrong_pose_has_higher_freespace_violation_than_truth() -> None:
    maze = load_map(MAP_PATH)
    cfg = LidarConfig(num_rays=360, range_min=0.05, range_max=4.0)
    true_pose = Pose2D(0.0, 0.0, 0.0)
    wrong_pose = Pose2D(1.5, -1.0, math.pi / 2)
    scan = LidarSimulator(maze, cfg).scan(true_pose)

    lookup = FastOccupancyLookup(maze)
    points = scan_to_points(scan, range_min=cfg.range_min, range_max=cfg.range_max)
    scorer = PoseScorer(
        lookup,
        np.asarray(points[::4], dtype=np.float64),
        (),
        (),
        hit_radius_cells=2,
    )

    assert scorer.freespace_violation_rate(true_pose) == 0.0
    assert scorer.freespace_violation_rate(wrong_pose) > 0.0


def test_impossible_pose_scores_zero() -> None:
    maze = load_map(MAP_PATH)
    cfg = LidarConfig(num_rays=360, range_min=0.05, range_max=4.0)
    true_pose = Pose2D(0.0, 0.0, 0.0)
    wrong_pose = Pose2D(1.5, -1.0, math.pi / 2)
    scan = LidarSimulator(maze, cfg).scan(true_pose)

    lookup = FastOccupancyLookup(maze)
    points = scan_to_points(scan, range_min=cfg.range_min, range_max=cfg.range_max)
    scorer = PoseScorer(
        lookup,
        np.asarray(points[::4], dtype=np.float64),
        (),
        (),
        hit_radius_cells=2,
    )

    assert scorer.score_fast(true_pose) > 0.5
    assert scorer.score_fast(wrong_pose) == 0.0


def test_true_pose_stays_plausible_with_freespace_gate() -> None:
    maze = load_map(MAP_PATH)
    cfg = LidarConfig(num_rays=360, range_min=0.05, range_max=4.0)
    localizer = InitialLocalizer(
        maze,
        InitialLocalizerConfig(
            min_match_score=0.35,
            use_grid_search=False,
            freespace_consistency=True,
        ),
    )
    true_pose = Pose2D(0.0, 0.0, 0.0)
    scan = LidarSimulator(maze, cfg).scan(true_pose)
    result = localizer.localize(scan, lidar_config=cfg)
    assert result.success
    assert result.pose is not None
