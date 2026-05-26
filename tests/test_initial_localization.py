import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from autolocalize import (
    InitialLocalizer,
    InitialLocalizerConfig,
    LidarConfig,
    LidarSimulator,
    Pose2D,
    load_map,
)
from autolocalize.features.map_features import extract_map_corners
from autolocalize.features.scan import extract_scan_corners
from autolocalize.geometry.transform import pose_error
from autolocalize.localization.hypotheses import generate_feature_hypotheses
from autolocalize.map.grid import CellState, OccupancyGrid

MAP_PATH = Path(__file__).resolve().parents[1] / "maps" / "churchsidemaze1.yaml"

# Fixed maze poses (x, y, theta) — corridors, corners, near walls.
MAZE_FIXED_POSES = [
    (0.0, 0.0, 0.0),
    (-1.0, -1.0, math.pi / 4),
    (1.0, 0.5, -math.pi / 3),
    (0.5, -1.5, math.pi / 2),
    (1.5, 1.5, math.pi),
    (-0.5, 2.0, math.pi / 3),
    (0.5, 1.0, -math.pi / 4),
    (-0.8, -0.8, 0.2),
    (1.2, -1.0, 1.0),
    (-1.2, 1.2, 2.0),
]


def _box_map(size_m: float = 4.0, res: float = 0.05) -> OccupancyGrid:
    """Square room with walls on the boundary, free interior."""
    n = int(size_m / res)
    cells = [CellState.FREE] * (n * n)
    for gy in range(n):
        for gx in range(n):
            if gx == 0 or gy == 0 or gx == n - 1 or gy == n - 1:
                cells[gy * n + gx] = CellState.OCCUPIED
    return OccupancyGrid(
        width=n,
        height=n,
        resolution=res,
        origin_x=0.0,
        origin_y=0.0,
        origin_yaw=0.0,
        cells=tuple(cells),
    )


def _free_world_positions(grid: OccupancyGrid) -> list[tuple[float, float]]:
    positions: list[tuple[float, float]] = []
    for gy in range(grid.height):
        for gx in range(grid.width):
            if grid.cell_at(gx, gy) == CellState.FREE:
                positions.append(grid.grid_to_world_center(gx, gy))
    return positions


def _sample_maze_poses(
    grid: OccupancyGrid, *, count: int, seed: int
) -> list[Pose2D]:
    rng = random.Random(seed)
    free = _free_world_positions(grid)
    return [
        Pose2D(xy[0], xy[1], rng.uniform(-math.pi, math.pi))
        for xy in (rng.choice(free) for _ in range(count))
    ]


@dataclass(frozen=True, slots=True)
class MazeBatchResult:
    success: int
    failed: int
    skipped_no_corners: int
    median_localize_s: float
    total_localize_s: float


def _run_maze_localization_batch(
    maze: OccupancyGrid,
    lidar_cfg: LidarConfig,
    localizer_cfg: InitialLocalizerConfig,
    *,
    count: int,
    seed: int,
    trans_tol: float = 0.3,
    rot_tol: float = 0.4,
) -> MazeBatchResult:
    sim = LidarSimulator(maze, lidar_cfg)
    localizer = InitialLocalizer(maze, localizer_cfg)
    _ = localizer.map_corners

    success = 0
    failed = 0
    skipped_no_corners = 0
    times: list[float] = []

    for true_pose in _sample_maze_poses(maze, count=count, seed=seed):
        scan = sim.scan(true_pose)
        scan_corners = extract_scan_corners(
            scan,
            range_min=lidar_cfg.range_min,
            range_max=lidar_cfg.range_max,
        )
        if not scan_corners:
            skipped_no_corners += 1
            continue

        t0 = time.perf_counter()
        result = localizer.localize(scan, lidar_config=lidar_cfg)
        times.append(time.perf_counter() - t0)

        if not result.success or result.pose is None:
            failed += 1
            continue

        trans_err, rot_err = pose_error(result.pose, true_pose)
        if trans_err < trans_tol and rot_err < rot_tol:
            success += 1
        else:
            failed += 1

    median_s = sorted(times)[len(times) // 2] if times else 0.0
    return MazeBatchResult(
        success=success,
        failed=failed,
        skipped_no_corners=skipped_no_corners,
        median_localize_s=median_s,
        total_localize_s=sum(times),
    )


def _pose_in_hypotheses(
    hypotheses: list[Pose2D],
    truth: Pose2D,
    *,
    trans_tol: float = 0.12,
    rot_tol: float = 0.15,
) -> bool:
    for pose in hypotheses:
        trans_err, rot_err = pose_error(pose, truth)
        if trans_err < trans_tol and rot_err < rot_tol:
            return True
    return False


@pytest.fixture
def box() -> OccupancyGrid:
    return _box_map()


@pytest.fixture
def maze():
    return load_map(MAP_PATH)


@pytest.fixture
def lidar_cfg() -> LidarConfig:
    return LidarConfig(num_rays=360, range_min=0.05, range_max=4.0)


@pytest.fixture
def localizer_cfg() -> InitialLocalizerConfig:
    return InitialLocalizerConfig(min_match_score=0.35, use_grid_search=False)


def test_map_corners_on_box(box: OccupancyGrid) -> None:
    corners = extract_map_corners(box)
    assert len(corners) == 4


def test_scan_corners_in_box_room(box: OccupancyGrid, lidar_cfg: LidarConfig) -> None:
    sim = LidarSimulator(box, lidar_cfg)
    true_pose = Pose2D(2.0, 2.0, 0.0)
    scan = sim.scan(true_pose)
    corners = extract_scan_corners(
        scan,
        range_min=lidar_cfg.range_min,
        range_max=lidar_cfg.range_max,
    )
    assert len(corners) >= 3


def test_true_pose_in_feature_hypotheses_maze(
    maze: OccupancyGrid, lidar_cfg: LidarConfig
) -> None:
    """Feature enumeration must include the ground-truth pose (no grid search)."""
    sim = LidarSimulator(maze, lidar_cfg)
    true_pose = Pose2D(0.0, 0.0, 0.0)
    scan = sim.scan(true_pose)

    scan_corners = extract_scan_corners(
        scan,
        range_min=lidar_cfg.range_min,
        range_max=lidar_cfg.range_max,
    )
    map_corners = extract_map_corners(maze)
    hypotheses, _ = generate_feature_hypotheses(
        scan_corners,
        map_corners,
        grid_resolution=maze.resolution,
    )

    assert _pose_in_hypotheses(hypotheses, true_pose), (
        f"true pose missing among {len(hypotheses)} feature hypotheses "
        f"({len(scan_corners)} scan corners, {len(map_corners)} map corners)"
    )


def test_localize_in_box_room(box: OccupancyGrid, lidar_cfg: LidarConfig) -> None:
    sim = LidarSimulator(box, lidar_cfg)
    true_pose = Pose2D(1.3, 1.7, math.pi / 6)

    scan = sim.scan(true_pose)
    localizer = InitialLocalizer(
        box,
        InitialLocalizerConfig(
            min_match_score=0.5,
            use_grid_search=False,
            refine_top_k=1,
        ),
    )
    result = localizer.localize(scan, lidar_config=lidar_cfg)

    assert result.success, (
        f"localization failed score={result.score:.3f} "
        f"scan_corners={len(result.scan_corners)} "
        f"hypotheses={result.hypotheses_tested}"
    )
    assert result.pose is not None
    trans_err, rot_err = pose_error(result.pose, true_pose)
    assert trans_err < 0.15, f"translation error {trans_err:.3f} m"
    assert rot_err < 0.2, f"rotation error {rot_err:.3f} rad"


def test_localize_maze_known_pose(
    maze: OccupancyGrid, lidar_cfg: LidarConfig, localizer_cfg: InitialLocalizerConfig
) -> None:
    sim = LidarSimulator(maze, lidar_cfg)
    true_pose = Pose2D(0.0, 0.0, 0.0)

    scan = sim.scan(true_pose)
    localizer = InitialLocalizer(maze, localizer_cfg)
    result = localizer.localize(scan, lidar_config=lidar_cfg)

    assert result.success, (
        f"score={result.score:.3f} corners={len(result.scan_corners)} "
        f"hyp={result.hypotheses_tested}"
    )
    trans_err, rot_err = pose_error(result.pose, true_pose)
    assert trans_err < 0.25
    assert rot_err < 0.35


@pytest.mark.parametrize("x,y,theta", MAZE_FIXED_POSES)
def test_maze_fixed_poses(
    maze: OccupancyGrid,
    lidar_cfg: LidarConfig,
    localizer_cfg: InitialLocalizerConfig,
    x: float,
    y: float,
    theta: float,
) -> None:
    sim = LidarSimulator(maze, lidar_cfg)
    true_pose = Pose2D(x, y, theta)
    scan = sim.scan(true_pose)

    localizer = InitialLocalizer(maze, localizer_cfg)
    result = localizer.localize(scan, lidar_config=lidar_cfg)
    assert result.success, f"pose ({x},{y},{theta}): score={result.score:.3f}"
    trans_err, rot_err = pose_error(result.pose, true_pose)
    assert trans_err < 0.3, f"pose ({x},{y},{theta}): trans={trans_err}"
    assert rot_err < 0.4


@pytest.mark.slow
def test_maze_random_poses(
    maze: OccupancyGrid,
    lidar_cfg: LidarConfig,
    localizer_cfg: InitialLocalizerConfig,
) -> None:
    """30 random free-space poses; expect high success rate."""
    batch = _run_maze_localization_batch(
        maze, lidar_cfg, localizer_cfg, count=30, seed=42
    )
    assert batch.failed <= 3, (
        f"failures={batch.failed} success={batch.success} "
        f"skipped={batch.skipped_no_corners}"
    )


@pytest.mark.slow
@pytest.mark.stress
def test_maze_random_poses_1000(
    maze: OccupancyGrid,
    lidar_cfg: LidarConfig,
    localizer_cfg: InitialLocalizerConfig,
) -> None:
    """1000 random free-space poses — Monte Carlo regression on maze map."""
    batch = _run_maze_localization_batch(
        maze, lidar_cfg, localizer_cfg, count=1000, seed=42
    )
    evaluated = batch.success + batch.failed
    assert evaluated > 0
    success_rate = batch.success / evaluated
    assert success_rate >= 0.99, (
        f"success {batch.success}/{evaluated} ({success_rate:.1%}), "
        f"skipped_no_corners={batch.skipped_no_corners}, "
        f"median={batch.median_localize_s * 1000:.0f}ms "
        f"total={batch.total_localize_s:.1f}s"
    )
    assert batch.median_localize_s < 1.0, (
        f"median localize {batch.median_localize_s * 1000:.0f}ms too slow"
    )


def test_maze_localization_speed(
    maze: OccupancyGrid,
    lidar_cfg: LidarConfig,
    localizer_cfg: InitialLocalizerConfig,
) -> None:
    """Ten random poses should localize quickly (median; allows CI runner variance)."""
    sim = LidarSimulator(maze, lidar_cfg)
    localizer = InitialLocalizer(maze, localizer_cfg)
    _ = localizer.map_corners

    times: list[float] = []
    for true_pose in _sample_maze_poses(maze, count=10, seed=7):
        scan = sim.scan(true_pose)
        t0 = time.perf_counter()
        result = localizer.localize(scan, lidar_config=lidar_cfg)
        times.append(time.perf_counter() - t0)
        assert result.success

    median_s = sorted(times)[len(times) // 2]
    assert median_s < 1.2, f"median localize time {median_s:.3f}s too slow"


def test_greedy_tries_fewer_than_full_enumeration(
    maze: OccupancyGrid, lidar_cfg: LidarConfig, localizer_cfg: InitialLocalizerConfig
) -> None:
    sim = LidarSimulator(maze, lidar_cfg)
    scan = sim.scan(Pose2D(0.0, 0.0, 0.0))
    localizer = InitialLocalizer(maze, localizer_cfg)
    result = localizer.localize(scan, lidar_config=lidar_cfg)

    scan_corners = extract_scan_corners(
        scan,
        range_min=lidar_cfg.range_min,
        range_max=lidar_cfg.range_max,
    )
    map_corners = extract_map_corners(maze)
    full_count, _ = generate_feature_hypotheses(
        scan_corners,
        map_corners,
        grid_resolution=maze.resolution,
    )

    assert result.success
    assert result.hypotheses_tested < len(full_count) // 2, (
        f"greedy tried {result.hypotheses_tested} vs {len(full_count)} full enumeration"
    )


def test_wrong_pose_scores_lower(box: OccupancyGrid, lidar_cfg: LidarConfig) -> None:
    from autolocalize.localization.scoring import score_scan_against_map

    sim = LidarSimulator(box, lidar_cfg)
    true_pose = Pose2D(2.0, 2.0, 0.5)
    scan = sim.scan(true_pose)

    good = score_scan_against_map(
        maze := box, scan, true_pose, range_max=lidar_cfg.range_max
    )
    bad = score_scan_against_map(
        maze, scan, Pose2D(0.5, 0.5, -1.0), range_max=lidar_cfg.range_max
    )
    assert good > bad + 0.2
