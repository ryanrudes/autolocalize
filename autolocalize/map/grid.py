from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto


class CellState(Enum):
    FREE = auto()
    OCCUPIED = auto()
    UNKNOWN = auto()


@dataclass(frozen=True, slots=True)
class OccupancyGrid:
    """ROS map_server-style occupancy grid (cell 0,0 = bottom-left in world)."""

    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    origin_yaw: float
    cells: tuple[CellState, ...]

    def __post_init__(self) -> None:
        expected = self.width * self.height
        if len(self.cells) != expected:
            raise ValueError(
                f"cells length {len(self.cells)} != width*height ({expected})"
            )

    def index(self, gx: int, gy: int) -> int:
        return gy * self.width + gx

    def in_bounds(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self.width and 0 <= gy < self.height

    def cell_at(self, gx: int, gy: int) -> CellState:
        return self.cells[self.index(gx, gy)]

    def is_blocking(self, gx: int, gy: int) -> bool:
        if not self.in_bounds(gx, gy):
            return True
        state = self.cell_at(gx, gy)
        return state in (CellState.OCCUPIED, CellState.UNKNOWN)

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        gx = math.floor((x - self.origin_x) / self.resolution)
        gy = math.floor((y - self.origin_y) / self.resolution)
        return gx, gy

    def grid_to_world_center(self, gx: int, gy: int) -> tuple[float, float]:
        x = self.origin_x + (gx + 0.5) * self.resolution
        y = self.origin_y + (gy + 0.5) * self.resolution
        return x, y

    @property
    def world_width(self) -> float:
        return self.width * self.resolution

    @property
    def world_height(self) -> float:
        return self.height * self.resolution
