import math
from pathlib import Path

import pytest

from autolocalize import LidarConfig, LidarSimulator, Pose2D, load_map
from autolocalize.map.grid import CellState, OccupancyGrid
from autolocalize.sensors.raycast import cast_ray

MAP_PATH = Path(__file__).resolve().parents[1] / "maps" / "churchsidemaze1.yaml"


@pytest.fixture
def maze() -> OccupancyGrid:
    return load_map(MAP_PATH)


@pytest.fixture
def open_grid() -> OccupancyGrid:
    return OccupancyGrid(
        width=10,
        height=10,
        resolution=0.1,
        origin_x=0.0,
        origin_y=0.0,
        origin_yaw=0.0,
        cells=tuple(CellState.FREE for _ in range(100)),
    )


def test_open_space_ray_hits_range_max(open_grid: OccupancyGrid) -> None:
    r = cast_ray(open_grid, 0.5, 0.5, 0.0, range_min=0.05, range_max=2.0)
    assert r == pytest.approx(2.0)


def test_wall_hit_distance(open_grid: OccupancyGrid) -> None:
    cells = list(open_grid.cells)
    for gy in range(10):
        cells[gy * 10 + 9] = CellState.OCCUPIED
    walled = OccupancyGrid(
        width=10,
        height=10,
        resolution=0.1,
        origin_x=0.0,
        origin_y=0.0,
        origin_yaw=0.0,
        cells=tuple(cells),
    )
    r = cast_ray(walled, 0.5, 0.5, 0.0, range_min=0.05, range_max=5.0)
    assert 0.4 < r < 0.6


def test_lidar_scan_shape(maze: OccupancyGrid) -> None:
    sim = LidarSimulator(maze, LidarConfig(num_rays=720, range_max=3.0))
    scan = sim.scan(Pose2D(0.0, 0.0, 0.0))
    assert len(scan.ranges) == 720
    assert len(scan.angles) == 720
    assert all(0.05 <= d <= 3.0 for d in scan.ranges)


def test_lidar_full_circle_coverage(maze: OccupancyGrid) -> None:
    cfg = LidarConfig(num_rays=4, angle_min=-math.pi, angle_max=math.pi)
    sim = LidarSimulator(maze, cfg)
    angles = sim.config.beam_angles()
    assert len(angles) == 4
    assert angles[0] == pytest.approx(-math.pi)
    assert angles[-1] == pytest.approx(math.pi)
