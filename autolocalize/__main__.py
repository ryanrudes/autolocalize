"""CLI demo: localize one simulated scan on the bundled maze map."""

from __future__ import annotations

from pathlib import Path

from autolocalize import (
    InitialLocalizer,
    InitialLocalizerConfig,
    LidarConfig,
    LidarSimulator,
    Pose2D,
    load_map,
)
from autolocalize.geometry.transform import pose_error


def default_map_yaml() -> Path:
    """Resolve the sample map in a dev checkout or an installed wheel."""
    pkg_root = Path(__file__).resolve().parent
    for candidate in (
        pkg_root.parent / "maps" / "churchsidemaze1.yaml",
        pkg_root / "maps" / "churchsidemaze1.yaml",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "churchsidemaze1.yaml not found; run from the repo root or reinstall the package"
    )


def main() -> None:
    grid = load_map(default_map_yaml())

    lidar_cfg = LidarConfig(num_rays=720, range_min=0.05, range_max=5.0)
    sim = LidarSimulator(grid, lidar_cfg)

    true_pose = Pose2D(x=0.0, y=0.0, theta=0.0)
    scan = sim.scan(true_pose)

    localizer = InitialLocalizer(grid, InitialLocalizerConfig())
    result = localizer.localize(scan, lidar_config=lidar_cfg)

    print(f"Map: {grid.width}x{grid.height} @ {grid.resolution:.4f} m/cell")
    print(f"True pose: ({true_pose.x}, {true_pose.y}, {true_pose.theta:.3f} rad)")
    print(
        f"Features: {len(result.scan_corners)} scan corners, "
        f"{len(result.map_corners)} map corners"
    )
    print(f"Hypotheses tested: {result.hypotheses_tested}")

    if result.success and result.pose is not None:
        trans_err, rot_err = pose_error(result.pose, true_pose)
        print(
            f"Estimated pose: ({result.pose.x:.3f}, {result.pose.y:.3f}, "
            f"{result.pose.theta:.3f} rad)"
        )
        print(f"Match score: {result.score:.3f}")
        print(f"Error: {trans_err:.3f} m translation, {rot_err:.3f} rad rotation")
    else:
        print(f"Localization failed (best score {result.score:.3f})")


if __name__ == "__main__":
    main()
