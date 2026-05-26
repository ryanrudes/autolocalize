## Learned User Preferences

- Prefer feature-based initial localization over free-space grid search when accuracy allows; precompute map features for reuse.
- Expect feature-based localization to be fast on typical poses (often tens of ms with early exit); use `config_for_effort("adaptive")` for tiered effort (tiers 1–3) when balancing speed vs near-100% maze accuracy.
- Do not enumerate all combinatorial corner pairings before scoring; evaluate and prune hypotheses greedily.
- Target close to 100% localization success on maze maps; validate with large simulator batches (e.g. 1000 poses) or `scripts/live_benchmark.py`.
- Use the LIDAR simulator to generate robust automated tests for localization changes.
- Only create git commits when explicitly requested.

## Learned Workspace Facts

- Python 3.12+ uv project (`autolocalize` package) for automatic initial robot localization (LIDAR frame to map frame); published at https://github.com/ryanrudes/autolocalize with git remote named `GitHub` (not `origin`).
- Maps live under `maps/` as ROS occupancy grid YAML+PGM pairs (e.g. `churchsidemaze1.yaml`).
- `LidarSimulator` ray-casts against the occupancy grid; default config is 720 rays over 360° with configurable min/max range.
- Localization matches wall-corner features from map and LIDAR scan endpoints, refines candidates, and picks best pose; effort presets via `config_for_effort("standard"|"fast"|"adaptive")` — fast skips refine and limits corner pairs; adaptive escalates tiers 1–3 (refine-before-accept, no unrefined early exit).
- Connected-freespace checks (`autolocalize/map/freespace.py`) hard-reject impossible poses; hypothesis search ranks without per-ray freespace gating; final selection tie-breaks with corner-assignment cost.
- Tests are pytest-based in `tests/` (`slow`/`stress` markers for large batches); `scripts/visualize_localization.py` demos poses (`uv sync --extra viz`); `scripts/live_benchmark.py` provides a Rich live benchmark (`-n`, `--adaptive`, `--fast`, `uv sync --dev` for `rich`).
- CI `test_maze_localization_speed` uses median localize time < 1.2s to tolerate GitHub Actions runner variance.
