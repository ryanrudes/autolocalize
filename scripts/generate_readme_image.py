#!/usr/bin/env python3
"""
Render a static README figure: true vs estimated pose on the sample maze map.

Usage:
    uv sync --extra viz
    uv run python scripts/generate_readme_image.py
    uv run python scripts/generate_readme_image.py -o docs/readme-localization.png
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrow

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
from scripts.visualize_localization import (
    draw_pose_arrow,
    format_pose,
    occupancy_image,
    plot_map,
    plot_polar_scan,
    plot_scan_on_map,
)


def render_readme_figure(
    grid,
    true_pose: Pose2D,
    scan,
    result,
    lidar_cfg: LidarConfig,
    *,
    output: Path,
    dpi: int = 150,
) -> None:
    scan_pts = scan_to_points(
        scan,
        range_min=lidar_cfg.range_min,
        range_max=lidar_cfg.range_max,
    )

    fig = plt.figure(figsize=(13, 5.2), facecolor="white")
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.15, 1.15, 0.85], wspace=0.22)

    ax_before = fig.add_subplot(gs[0, 0])
    plot_map(ax_before, grid, title="Simulated scan (ground truth)")
    plot_scan_on_map(ax_before, scan_pts, true_pose, color="#2563eb", label="LIDAR hits")
    draw_pose_arrow(ax_before, true_pose, color="#16a34a", label="true pose")
    ax_before.legend(loc="upper right", fontsize=8, framealpha=0.92)

    ax_after = fig.add_subplot(gs[0, 1])
    plot_map(ax_after, grid, title="Localized pose (estimate)")
    plot_scan_on_map(
        ax_after,
        scan_pts,
        true_pose,
        color="#94a3b8",
        alpha=0.35,
        label="scan at true pose",
    )
    draw_pose_arrow(ax_after, true_pose, color="#16a34a", label="true pose")

    subtitle = "autolocalize — single-scan initial localization"
    if result.success and result.pose is not None:
        plot_scan_on_map(
            ax_after,
            scan_pts,
            result.pose,
            color="#dc2626",
            label="scan at estimate",
        )
        draw_pose_arrow(ax_after, result.pose, color="#dc2626", label="estimated pose")
        trans_err, rot_err = pose_error(result.pose, true_pose)
        subtitle = (
            f"Match score {result.score:.2f} · "
            f"error {trans_err:.3f} m, {math.degrees(rot_err):.1f}°"
        )
    ax_after.legend(loc="upper right", fontsize=8, framealpha=0.92)

    ax_polar = fig.add_subplot(gs[0, 2], projection="polar")
    plot_polar_scan(ax_polar, scan, lidar_cfg, title="Scan (robot frame)")

    fig.suptitle(subtitle, fontsize=13, fontweight="bold", y=1.02)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate README localization figure.")
    parser.add_argument(
        "--map",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "maps" / "churchsidemaze1.yaml",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs" / "readme-localization.png",
    )
    parser.add_argument("--seed", type=int, default=7, help="Pose index seed (see tests)")
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grid = load_map(args.map)

    lidar_cfg = LidarConfig(num_rays=360, range_min=0.05, range_max=4.0)
    sim = LidarSimulator(grid, lidar_cfg)
    localizer = InitialLocalizer(grid, InitialLocalizerConfig(use_grid_search=False))

    # Same sampling as speed tests: seed 7, first pose — good corridor view.
    import random

    from autolocalize.map.grid import CellState

    rng = random.Random(args.seed)
    free = [
        grid.grid_to_world_center(gx, gy)
        for gy in range(grid.height)
        for gx in range(grid.width)
        if grid.cell_at(gx, gy) == CellState.FREE
    ]
    xy = rng.choice(free)
    true_pose = Pose2D(xy[0], xy[1], rng.uniform(-math.pi, math.pi))

    scan = sim.scan(true_pose)
    result = localizer.localize(scan, lidar_config=lidar_cfg)

    if not result.success or result.pose is None:
        raise SystemExit(f"Localization failed for README figure (score={result.score:.3f})")

    trans_err, rot_err = pose_error(result.pose, true_pose)
    print(f"True:  {format_pose(true_pose)}")
    print(f"Est:   {format_pose(result.pose)}")
    print(f"Error: {trans_err:.3f} m, {math.degrees(rot_err):.1f}°")

    render_readme_figure(
        grid,
        true_pose,
        scan,
        result,
        lidar_cfg,
        output=args.output,
        dpi=args.dpi,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
