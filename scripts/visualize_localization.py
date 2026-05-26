#!/usr/bin/env python3
"""
Generate random starting poses, simulate LIDAR, run initial localization, and visualize.

Usage:
    uv sync --extra viz
    uv run python scripts/visualize_localization.py
    uv run python scripts/visualize_localization.py --count 5 --seed 1 --fast
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

# Allow running without installing the package in editable mode from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autolocalize import (
    InitialLocalizer,
    InitialLocalizerConfig,
    LidarConfig,
    LidarSimulator,
    Pose2D,
    load_map,
)
from autolocalize.features.scan import scan_to_points
from autolocalize.geometry.transform import apply_pose, pose_error
from autolocalize.map.grid import CellState, OccupancyGrid

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import FancyArrow
except ImportError:
    print(
        "matplotlib is required for visualization.\n"
        "Install with: uv sync --extra viz",
        file=sys.stderr,
    )
    sys.exit(1)


def sample_free_poses(
    grid: OccupancyGrid,
    count: int,
    rng: random.Random,
) -> list[Pose2D]:
    """Pick random poses on free cells with uniform random heading."""
    free_xy: list[tuple[float, float]] = []
    for gy in range(grid.height):
        for gx in range(grid.width):
            if grid.cell_at(gx, gy) == CellState.FREE:
                free_xy.append(grid.grid_to_world_center(gx, gy))

    if not free_xy:
        raise RuntimeError("Map has no free cells")

    poses: list[Pose2D] = []
    for _ in range(count):
        x, y = rng.choice(free_xy)
        theta = rng.uniform(-math.pi, math.pi)
        poses.append(Pose2D(x, y, theta))
    return poses


def occupancy_image(grid: OccupancyGrid) -> np.ndarray:
    """Raster for imshow: 0=occupied, 0.5=unknown, 1=free."""
    img = np.zeros((grid.height, grid.width), dtype=float)
    for gy in range(grid.height):
        for gx in range(grid.width):
            state = grid.cell_at(gx, gy)
            if state == CellState.FREE:
                img[gy, gx] = 1.0
            elif state == CellState.UNKNOWN:
                img[gy, gx] = 0.5
    return img


def plot_map(
    ax: plt.Axes,
    grid: OccupancyGrid,
    *,
    title: str,
) -> None:
    extent = (
        grid.origin_x,
        grid.origin_x + grid.world_width,
        grid.origin_y,
        grid.origin_y + grid.world_height,
    )
    ax.imshow(
        occupancy_image(grid),
        origin="lower",
        extent=extent,
        cmap="gray_r",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)


def draw_pose_arrow(
    ax: plt.Axes,
    pose: Pose2D,
    *,
    color: str,
    label: str,
    length: float = 0.35,
) -> None:
    dx = length * math.cos(pose.theta)
    dy = length * math.sin(pose.theta)
    ax.add_patch(
        FancyArrow(
            pose.x,
            pose.y,
            dx,
            dy,
            width=length * 0.35,
            color=color,
            label=label,
            length_includes_head=True,
        )
    )
    ax.plot(pose.x, pose.y, "o", color=color, markersize=6)


def plot_scan_on_map(
    ax: plt.Axes,
    scan_points_robot: tuple[tuple[float, float], ...],
    pose: Pose2D,
    *,
    color: str,
    alpha: float = 0.7,
    label: str,
) -> None:
    if not scan_points_robot:
        return
    wx: list[float] = []
    wy: list[float] = []
    for lx, ly in scan_points_robot:
        x, y = apply_pose(pose, lx, ly)
        wx.append(x)
        wy.append(y)
    ax.scatter(wx, wy, s=4, c=color, alpha=alpha, label=label, edgecolors="none")


def plot_polar_scan(
    ax: plt.Axes,
    scan,
    lidar_cfg: LidarConfig,
    *,
    title: str,
) -> None:
    ax.set_title(title)
    valid_angles: list[float] = []
    valid_ranges: list[float] = []
    for r, a in zip(scan.ranges, scan.angles):
        if r <= lidar_cfg.range_min or r >= lidar_cfg.range_max:
            continue
        valid_angles.append(a)
        valid_ranges.append(r)
    ax.scatter(valid_angles, valid_ranges, s=3, c="#2563eb")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, lidar_cfg.range_max)
    ax.set_xlabel("bearing (rad)")
    ax.grid(True, alpha=0.3)


def format_pose(pose: Pose2D) -> str:
    deg = math.degrees(pose.theta)
    return f"({pose.x:.3f}, {pose.y:.3f}, {pose.theta:.3f} rad / {deg:.1f}°)"


def show_trial(
    grid: OccupancyGrid,
    true_pose: Pose2D,
    scan,
    result,
    lidar_cfg: LidarConfig,
    *,
    index: int,
    total: int,
) -> None:
    scan_pts = scan_to_points(
        scan,
        range_min=lidar_cfg.range_min,
        range_max=lidar_cfg.range_max,
    )

    fig = plt.figure(figsize=(14, 7), layout="constrained")
    fig.suptitle(f"Trial {index + 1} / {total}", fontsize=14, fontweight="bold")
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.2, 1.0], hspace=0.3, wspace=0.25)

    ax_true = fig.add_subplot(gs[0, 0])
    plot_map(ax_true, grid, title="Ground truth: pose + LIDAR (map frame)")
    plot_scan_on_map(ax_true, scan_pts, true_pose, color="#2563eb", label="scan hits")
    draw_pose_arrow(ax_true, true_pose, color="#16a34a", label="true pose")
    ax_true.legend(loc="upper right", fontsize=8)

    ax_polar = fig.add_subplot(gs[0, 1], projection="polar")
    plot_polar_scan(ax_polar, scan, lidar_cfg, title="LIDAR scan (robot frame)")

    ax_est = fig.add_subplot(gs[1, 0])
    plot_map(ax_est, grid, title="After localization")
    plot_scan_on_map(
        ax_est,
        scan_pts,
        true_pose,
        color="#94a3b8",
        alpha=0.35,
        label="scan (at true pose)",
    )
    draw_pose_arrow(ax_est, true_pose, color="#16a34a", label="true pose")

    lines: list[str] = [
        f"True pose: {format_pose(true_pose)}",
        f"Scan rays used: {len(scan_pts)} / {scan.num_rays}",
        f"Scan corners: {len(result.scan_corners)}",
        f"Hypotheses tested: {result.hypotheses_tested}",
        "",
    ]

    if result.success and result.pose is not None:
        plot_scan_on_map(
            ax_est,
            scan_pts,
            result.pose,
            color="#dc2626",
            label="scan (at estimate)",
        )
        draw_pose_arrow(ax_est, result.pose, color="#dc2626", label="estimated pose")
        trans_err, rot_err = pose_error(result.pose, true_pose)
        lines.extend(
            [
                f"Estimated: {format_pose(result.pose)}",
                f"Match score: {result.score:.3f}",
                f"Error: {trans_err:.3f} m, {math.degrees(rot_err):.1f}°",
            ]
        )
        status = "SUCCESS" if trans_err < 0.3 else "SUCCESS (coarse)"
        status_color = "#16a34a" if trans_err < 0.3 else "#ca8a04"
    else:
        lines.extend(
            [
                "Localization failed.",
                f"Best score: {result.score:.3f}",
            ]
        )
        status = "FAILED"
        status_color = "#dc2626"

    ax_est.legend(loc="upper right", fontsize=8)

    ax_text = fig.add_subplot(gs[1, 1])
    ax_text.axis("off")
    ax_text.text(
        0.0,
        0.95,
        status,
        transform=ax_text.transAxes,
        fontsize=20,
        fontweight="bold",
        color=status_color,
        va="top",
    )
    ax_text.text(
        0.0,
        0.82,
        "\n".join(lines),
        transform=ax_text.transAxes,
        fontsize=11,
        family="monospace",
        va="top",
    )
    ax_text.text(
        0.0,
        0.05,
        "Close window or press any key for next trial.",
        transform=ax_text.transAxes,
        fontsize=9,
        color="#64748b",
        va="bottom",
    )

    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize LIDAR simulation and initial localization trials.",
    )
    parser.add_argument(
        "--map",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "maps" / "churchsidemaze1.yaml",
        help="Path to map YAML",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Number of random starting poses to generate",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed")
    parser.add_argument(
        "--rays",
        type=int,
        default=360,
        help="Number of LIDAR rays per scan",
    )
    parser.add_argument(
        "--range-max",
        type=float,
        default=4.0,
        help="LIDAR maximum range (m)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use a faster (less thorough) localizer config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    print(f"Loading map: {args.map}")
    grid = load_map(args.map)

    lidar_cfg = LidarConfig(
        num_rays=args.rays,
        range_min=0.05,
        range_max=args.range_max,
    )
    sim = LidarSimulator(grid, lidar_cfg)

    if args.fast:
        loc_cfg = InitialLocalizerConfig(
            use_grid_search=False,
            score_ray_stride=8,
            max_scan_corners_for_pairs=4,
            refine_poses=False,
        )
    else:
        loc_cfg = InitialLocalizerConfig(use_grid_search=False)
    localizer = InitialLocalizer(grid, loc_cfg)

    poses = sample_free_poses(grid, args.count, rng)
    print(f"Generated {len(poses)} poses (seed={args.seed})")

    for i, true_pose in enumerate(poses):
        print(f"\n--- Trial {i + 1}/{len(poses)} ---")
        print(f"True pose: {format_pose(true_pose)}")

        print("Simulating LIDAR...")
        scan = sim.scan(true_pose)

        print("Running initial localization...")
        result = localizer.localize(scan, lidar_config=lidar_cfg)

        if result.success and result.pose is not None:
            trans_err, rot_err = pose_error(result.pose, true_pose)
            print(f"Estimated: {format_pose(result.pose)}")
            print(f"Score: {result.score:.3f}  Error: {trans_err:.3f} m, {math.degrees(rot_err):.1f}°")
        else:
            print(f"Failed (best score {result.score:.3f})")

        show_trial(
            grid,
            true_pose,
            scan,
            result,
            lidar_cfg,
            index=i,
            total=len(poses),
        )


if __name__ == "__main__":
    main()
