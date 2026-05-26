from __future__ import annotations

import numpy as np

from autolocalize.map.grid import CellState, OccupancyGrid


def free_mask_from_grid(grid: OccupancyGrid) -> np.ndarray:
    """True where the cell is traversable free space."""
    mask = np.zeros((grid.height, grid.width), dtype=bool)
    for gy in range(grid.height):
        row = gy * grid.width
        for gx in range(grid.width):
            if grid.cells[row + gx] == CellState.FREE:
                mask[gy, gx] = True
    return mask


def label_free_components(free_mask: np.ndarray) -> np.ndarray:
    """
    Connected-component labels for free cells.

    Returns int32 array shaped like ``free_mask``; -1 for non-free cells.
    """
    h, w = free_mask.shape
    labels = np.full((h, w), -1, dtype=np.int32)
    label = 0

    for gy in range(h):
        for gx in range(w):
            if not free_mask[gy, gx] or labels[gy, gx] >= 0:
                continue
            stack = [(gx, gy)]
            labels[gy, gx] = label
            while stack:
                cx, cy = stack.pop()
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < w and 0 <= ny < h and free_mask[ny, nx] and labels[ny, nx] < 0:
                        labels[ny, nx] = label
                        stack.append((nx, ny))
            label += 1

    return labels


def binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Morphological dilation on a 2D bool grid."""
    if radius <= 0:
        return mask
    h, w = mask.shape
    out = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            y_out = slice(max(0, -dy), h - max(0, dy))
            x_out = slice(max(0, -dx), w - max(0, dx))
            y_src = slice(max(0, dy), h - max(0, -dy))
            x_src = slice(max(0, dx), w - max(0, -dx))
            out[y_out, x_out] |= mask[y_src, x_src]
    return out


class FreeSpaceIndex:
    """
    Connected free components and noise-expanded reachable regions.

    An endpoint is consistent with a pose when it lies in the robot's
    component or within ``noise_cells`` of that component (LIDAR noise
    outside the navigable boundary).
    """

    def __init__(self, grid: OccupancyGrid, *, noise_cells: int = 2) -> None:
        self.free_mask = free_mask_from_grid(grid)
        self.component_labels = label_free_components(self.free_mask)
        max_label = int(self.component_labels.max())
        self.num_components = max_label + 1 if max_label >= 0 else 0
        self._reachable: dict[int, np.ndarray] = {}
        for component_id in range(self.num_components):
            component = self.component_labels == component_id
            self._reachable[component_id] = binary_dilate(component, noise_cells)

    def robot_component(self, gx: int, gy: int) -> int | None:
        if not (0 <= gx < self.component_labels.shape[1] and 0 <= gy < self.component_labels.shape[0]):
            return None
        label = int(self.component_labels[gy, gx])
        return label if label >= 0 else None

    def reachable_region(self, component_id: int) -> np.ndarray:
        """Cells inside the component plus up to noise_cells outside its boundary."""
        return self._reachable[component_id]
