#!/usr/bin/env python3
"""
Live Monte Carlo localization benchmark with a Rich progress UI.

    uv sync --dev
    uv run python scripts/live_benchmark.py -n 100
    uv run python scripts/live_benchmark.py -n 1000 --seed 42 --adaptive
    uv run python scripts/live_benchmark.py -n 1000 --adaptive --icp
    uv run python scripts/live_benchmark.py -n 500 --fast
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table
    from rich.text import Text
except ImportError:
    print("Install rich:  uv pip install rich", file=sys.stderr)
    raise SystemExit(1) from None

from autolocalize import (
    InitialLocalizer,
    InitialLocalizerConfig,
    LidarConfig,
    LidarSimulator,
    Pose2D,
    load_map,
)
from autolocalize.localization.config import config_for_effort
from autolocalize.features.scan import extract_scan_corners
from autolocalize.geometry.transform import pose_error
from autolocalize.map.grid import CellState, OccupancyGrid

DEFAULT_MAP = Path(__file__).resolve().parents[1] / "maps" / "churchsidemaze1.yaml"


@dataclass
class RunStats:
    total: int
    done: int = 0
    success: int = 0
    failed: int = 0
    skipped_no_corners: int = 0
    icp_refined: int = 0
    localize_times_s: list[float] = field(default_factory=list)
    last_error: str = ""
    started_at: float = field(default_factory=time.perf_counter)

    @property
    def evaluated(self) -> int:
        return self.success + self.failed

    @property
    def success_pct(self) -> float:
        if self.evaluated == 0:
            return 0.0
        return 100.0 * self.success / self.evaluated

    @property
    def failure_pct(self) -> float:
        if self.evaluated == 0:
            return 0.0
        return 100.0 * self.failed / self.evaluated

    @property
    def median_ms(self) -> float:
        if not self.localize_times_s:
            return 0.0
        sorted_t = sorted(self.localize_times_s)
        return sorted_t[len(sorted_t) // 2] * 1000.0

    @property
    def elapsed_s(self) -> float:
        return time.perf_counter() - self.started_at


def sample_poses(grid: OccupancyGrid, *, count: int, seed: int) -> list[Pose2D]:
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


def build_stats_table(stats: RunStats, *, trans_tol: float, rot_tol: float) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", justify="right")
    table.add_column()

    table.add_row("Trials", f"{stats.done} / {stats.total}")
    table.add_row(
        "Success",
        Text(
            f"{stats.success}  ({stats.success_pct:.1f}%)",
            style="bold green" if stats.success_pct >= 95 else "yellow",
        ),
    )
    table.add_row(
        "Failed",
        Text(
            f"{stats.failed}  ({stats.failure_pct:.1f}%)",
            style="bold red" if stats.failed else "dim",
        ),
    )
    table.add_row("Skipped (no corners)", str(stats.skipped_no_corners))
    if stats.icp_refined:
        table.add_row("ICP refined", str(stats.icp_refined))
    table.add_row("Median localize", f"{stats.median_ms:.0f} ms")
    table.add_row("Elapsed", f"{stats.elapsed_s:.1f} s")
    table.add_row("Tolerance", f"{trans_tol:.2f} m trans, {math.degrees(rot_tol):.0f}° rot")
    if stats.last_error:
        table.add_row("Last issue", Text(stats.last_error, style="dim red"))
    return table


def build_display(
    progress: Progress,
    stats: RunStats,
    *,
    trans_tol: float,
    rot_tol: float,
) -> Group:
    stats_panel = Panel(
        build_stats_table(stats, trans_tol=trans_tol, rot_tol=rot_tol),
        title="Localization benchmark",
        border_style="blue",
    )
    return Group(progress, stats_panel)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live maze localization Monte Carlo benchmark.")
    p.add_argument("-n", "--count", type=int, default=30, help="Number of random poses")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for poses")
    p.add_argument("--map", type=Path, default=DEFAULT_MAP, help="Map YAML path")
    p.add_argument("--rays", type=int, default=360, help="LIDAR rays per scan")
    p.add_argument("--trans-tol", type=float, default=0.3, help="Success translation (m)")
    p.add_argument("--rot-tol", type=float, default=0.4, help="Success rotation (rad)")
    p.add_argument("--fast", action="store_true", help="Fixed fast localizer config")
    p.add_argument(
        "--adaptive",
        action="store_true",
        help="Adaptive tiered localizer (fast when confident, deep when not)",
    )
    p.add_argument(
        "--icp",
        action="store_true",
        help="Optional post-refinement: point-to-line ICP (sub-cm accuracy)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()

    if args.count < 1:
        console.print("[red]--count must be >= 1[/red]")
        raise SystemExit(1)

    console.print(f"[dim]Loading map[/dim] {args.map}")
    grid = load_map(args.map)

    lidar_cfg = LidarConfig(num_rays=args.rays, range_min=0.05, range_max=4.0)
    if args.fast and args.adaptive:
        console.print("[red]Use only one of --fast or --adaptive[/red]")
        raise SystemExit(1)
    if args.adaptive:
        loc_cfg = config_for_effort("adaptive")
    elif args.fast:
        loc_cfg = config_for_effort("fast")
    else:
        loc_cfg = InitialLocalizerConfig(use_grid_search=False)

    if args.icp:
        loc_cfg = replace(loc_cfg, refine_icp=True, icp_ray_stride=1)

    sim = LidarSimulator(grid, lidar_cfg)
    localizer = InitialLocalizer(grid, loc_cfg)
    _ = localizer.map_corners

    poses = sample_poses(grid, count=args.count, seed=args.seed)
    stats = RunStats(total=args.count)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        expand=False,
    )
    task_id = progress.add_task("Localizing", total=args.count)

    with Live(
        build_display(progress, stats, trans_tol=args.trans_tol, rot_tol=args.rot_tol),
        console=console,
        refresh_per_second=12,
    ) as live:
        for true_pose in poses:
            scan = sim.scan(true_pose)
            corners = extract_scan_corners(
                scan,
                range_min=lidar_cfg.range_min,
                range_max=lidar_cfg.range_max,
            )
            if not corners:
                stats.skipped_no_corners += 1
                stats.done += 1
                progress.advance(task_id)
                live.update(
                    build_display(
                        progress, stats, trans_tol=args.trans_tol, rot_tol=args.rot_tol
                    )
                )
                continue

            t0 = time.perf_counter()
            result = localizer.localize(scan, lidar_config=lidar_cfg)
            stats.localize_times_s.append(time.perf_counter() - t0)
            stats.done += 1

            if not result.success or result.pose is None:
                stats.failed += 1
                stats.last_error = f"no pose (score {result.score:.3f})"
            else:
                trans_err, rot_err = pose_error(result.pose, true_pose)
                if result.icp_refined:
                    stats.icp_refined += 1
                if trans_err < args.trans_tol and rot_err < args.rot_tol:
                    stats.success += 1
                    stats.last_error = ""
                else:
                    stats.failed += 1
                    stats.last_error = (
                        f"err {trans_err:.2f} m, {math.degrees(rot_err):.0f}°"
                    )

            progress.advance(task_id)
            live.update(
                build_display(
                    progress, stats, trans_tol=args.trans_tol, rot_tol=args.rot_tol
                )
            )

    console.print()
    if stats.evaluated:
        color = (
            "green"
            if stats.success_pct >= 99
            else "yellow"
            if stats.success_pct >= 95
            else "red"
        )
        console.print(
            f"[bold {color}]Done: {stats.success}/{stats.evaluated} passed "
            f"({stats.success_pct:.1f}% success, {stats.failure_pct:.1f}% failed)"
            f"[/bold {color}]"
        )
    if stats.skipped_no_corners:
        console.print(f"[dim]Skipped {stats.skipped_no_corners} (no scan corners)[/dim]")
    console.print(
        f"[dim]Median localize: {stats.median_ms:.0f} ms · total {stats.elapsed_s:.1f} s[/dim]"
    )


if __name__ == "__main__":
    main()
