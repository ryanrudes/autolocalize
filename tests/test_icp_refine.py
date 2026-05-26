"""Optional point-to-line ICP post-refinement."""

from __future__ import annotations

import math
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from autolocalize import InitialLocalizer, LidarConfig, LidarSimulator, Pose2D, load_map
from autolocalize.features.scan import scan_to_points
from autolocalize.geometry.transform import pose_error
from autolocalize.localization.config import InitialLocalizerConfig, config_for_effort
from autolocalize.localization.icp_refine import refine_pose_icp
from autolocalize.map.grid import CellState

MAP_PATH = Path(__file__).resolve().parents[1] / "maps" / "churchsidemaze1.yaml"


@pytest.fixture
def maze():
    return load_map(MAP_PATH)


def _sample_pose(maze, *, seed: int, index: int) -> Pose2D:
    rng = random.Random(seed)
    free = [
        maze.grid_to_world_center(gx, gy)
        for gy in range(maze.height)
        for gx in range(maze.width)
        if maze.cell_at(gx, gy) == CellState.FREE
    ]
    return [
        Pose2D(xy[0], xy[1], rng.uniform(-math.pi, math.pi))
        for xy in (rng.choice(free) for _ in range(index + 1))
    ][index]


def test_icp_disabled_by_default(maze) -> None:
    cfg = config_for_effort("adaptive")
    assert cfg.refine_icp is False


def test_icp_refines_perturbed_pose(maze) -> None:
    true_pose = _sample_pose(maze, seed=42, index=313)
    lidar = LidarConfig(num_rays=360, range_min=0.05, range_max=4.0)
    sim = LidarSimulator(maze, lidar)
    scan = sim.scan(true_pose)
    points = scan_to_points(scan, range_min=lidar.range_min, range_max=lidar.range_max)

    coarse = Pose2D(
        true_pose.x + 0.05,
        true_pose.y - 0.04,
        true_pose.theta + 0.12,
    )
    loc = InitialLocalizer(maze)
    cfg = replace(
        InitialLocalizerConfig(),
        refine_icp=True,
        icp_ray_stride=1,
    )
    icp = refine_pose_icp(
        loc.lookup,
        coarse,
        np.asarray(points, dtype=np.float64),
        cfg,
    )
    trans_before, rot_before = pose_error(coarse, true_pose)
    trans_after, rot_after = pose_error(icp.pose, true_pose)
    assert trans_after < trans_before
    assert rot_after < rot_before
    assert trans_after < 0.01
    assert rot_after < 0.02


def test_localize_with_icp_addon_improves_accuracy(maze) -> None:
    true_pose = _sample_pose(maze, seed=42, index=777)
    lidar = LidarConfig(num_rays=360, range_min=0.05, range_max=4.0)
    sim = LidarSimulator(maze, lidar)
    scan = sim.scan(true_pose)

    base_cfg = config_for_effort("adaptive")
    base = InitialLocalizer(maze, base_cfg).localize(scan, lidar_config=lidar)
    precise = InitialLocalizer(
        maze, replace(base_cfg, refine_icp=True, icp_ray_stride=1)
    ).localize(scan, lidar_config=lidar)

    assert base.success and base.pose is not None
    assert precise.success and precise.pose is not None
    assert precise.icp_refined
    assert precise.icp_mean_residual is not None

    base_err = pose_error(base.pose, true_pose)
    precise_err = pose_error(precise.pose, true_pose)
    assert precise_err[0] <= base_err[0] + 1e-6
    assert precise_err[1] <= base_err[1] + 1e-6
    assert precise_err[0] < 0.02
    assert precise_err[1] < 0.02


@pytest.mark.slow
def test_icp_batch_median_sub_centimeter(maze) -> None:
    rng = random.Random(7)
    free = [
        maze.grid_to_world_center(gx, gy)
        for gy in range(maze.height)
        for gx in range(maze.width)
        if maze.cell_at(gx, gy) == CellState.FREE
    ]
    poses = [
        Pose2D(xy[0], xy[1], rng.uniform(-math.pi, math.pi))
        for xy in (rng.choice(free) for _ in range(100))
    ]
    lidar = LidarConfig(num_rays=360, range_min=0.05, range_max=4.0)
    sim = LidarSimulator(maze, lidar)
    cfg = replace(config_for_effort("adaptive"), refine_icp=True, icp_ray_stride=1)
    loc = InitialLocalizer(maze, cfg)

    trans_errors: list[float] = []
    for true in poses:
        result = loc.localize(sim.scan(true), lidar_config=lidar)
        assert result.success and result.pose is not None
        trans_err, _ = pose_error(result.pose, true)
        trans_errors.append(trans_err)

    trans_errors.sort()
    assert trans_errors[len(trans_errors) // 2] < 0.01
