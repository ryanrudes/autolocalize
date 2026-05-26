"""Adaptive effort tiering: accuracy vs standard, faster on average."""

import math
import random
import time
from pathlib import Path

import pytest

from autolocalize import (
    InitialLocalizer,
    LidarConfig,
    LidarSimulator,
    Pose2D,
    load_map,
)
from autolocalize.features.scan import extract_scan_corners
from autolocalize.geometry.transform import pose_error
from autolocalize.localization.config import config_for_effort
from autolocalize.map.grid import CellState

MAP_PATH = Path(__file__).resolve().parents[1] / "maps" / "churchsidemaze1.yaml"


def _sample_poses(grid, *, count: int, seed: int) -> list[Pose2D]:
    rng = random.Random(seed)
    free = [
        grid.grid_to_world_center(gx, gy)
        for gy in range(grid.height)
        for gx in range(grid.width)
        if grid.cell_at(gx, gy) == CellState.FREE
    ]
    return [
        Pose2D(xy[0], xy[1], rng.uniform(-math.pi, math.pi))
        for xy in (rng.choice(free) for _ in range(count))
    ]


def _run_batch(grid, cfg, *, count: int, seed: int) -> dict[str, float | int]:
    lidar = LidarConfig(num_rays=360, range_min=0.05, range_max=4.0)
    sim = LidarSimulator(grid, lidar)
    loc = InitialLocalizer(grid, cfg)
    _ = loc.map_corners

    ok = failed = 0
    times: list[float] = []
    tiers: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}

    for true in _sample_poses(grid, count=count, seed=seed):
        scan = sim.scan(true)
        if not extract_scan_corners(
            scan, range_min=lidar.range_min, range_max=lidar.range_max
        ):
            continue
        t0 = time.perf_counter()
        result = loc.localize(scan, lidar_config=lidar)
        times.append(time.perf_counter() - t0)
        tier = result.effort_tier if result.effort_tier is not None else -1
        if tier in tiers:
            tiers[tier] += 1
        if not result.success or result.pose is None:
            failed += 1
            continue
        te, re = pose_error(result.pose, true)
        if te < 0.3 and re < 0.4:
            ok += 1
        else:
            failed += 1

    evaluated = ok + failed
    sorted_t = sorted(times)
    return {
        "ok": ok,
        "failed": failed,
        "evaluated": evaluated,
        "success_pct": 100.0 * ok / evaluated if evaluated else 0.0,
        "median_ms": sorted_t[len(sorted_t) // 2] * 1000 if sorted_t else 0.0,
        "mean_ms": (sum(times) / len(times) * 1000) if times else 0.0,
        "tier0": tiers[0],
        "tier1": tiers[1],
        "tier2": tiers[2],
        "tier3": tiers[3],
    }


@pytest.fixture
def maze():
    return load_map(MAP_PATH)


@pytest.mark.slow
def test_adaptive_matches_standard_accuracy(maze) -> None:
    """Adaptive should be near standard success on a few hundred poses."""
    n, seed = 400, 42
    adaptive = _run_batch(maze, config_for_effort("adaptive"), count=n, seed=seed)
    standard = _run_batch(maze, config_for_effort("standard"), count=n, seed=seed)

    assert adaptive["success_pct"] >= standard["success_pct"] - 1.0, (
        f"adaptive {adaptive['success_pct']:.1f}% vs standard {standard['success_pct']:.1f}%"
    )
    assert adaptive["success_pct"] >= 98.0


@pytest.mark.slow
def test_adaptive_faster_than_standard_median(maze) -> None:
    """Median time should beat standard (not pay tier overhead on every pose)."""
    n, seed = 400, 7
    adaptive = _run_batch(maze, config_for_effort("adaptive"), count=n, seed=seed)
    standard = _run_batch(maze, config_for_effort("standard"), count=n, seed=seed)

    assert adaptive["median_ms"] < standard["median_ms"] * 0.85, (
        f"adaptive median {adaptive['median_ms']:.0f}ms not faster than "
        f"standard {standard['median_ms']:.0f}ms"
    )


@pytest.mark.slow
def test_adaptive_beats_fast_accuracy(maze) -> None:
    n, seed = 400, 42
    adaptive = _run_batch(maze, config_for_effort("adaptive"), count=n, seed=seed)
    fast = _run_batch(maze, config_for_effort("fast"), count=n, seed=seed)

    assert adaptive["success_pct"] >= fast["success_pct"] + 5.0
    assert adaptive["median_ms"] < fast["median_ms"] * 2.5


@pytest.mark.slow
def test_adaptive_tier_distribution_not_all_tier3(maze) -> None:
    """Most poses should exit before the expensive tier."""
    stats = _run_batch(maze, config_for_effort("adaptive"), count=500, seed=99)
    deep = stats["tier3"]
    total = stats["tier0"] + stats["tier1"] + stats["tier2"] + stats["tier3"]
    assert total > 0
    assert deep / total < 0.25, f"too many tier-3 runs: {deep}/{total}"
