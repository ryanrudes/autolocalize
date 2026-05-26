## Learned User Preferences

- Prefer feature-based initial localization over free-space grid search when accuracy allows; precompute map features for reuse.
- Expect feature-based localization to be fast on typical poses (~70–115 ms median adaptive on maze, machine-dependent); use `config_for_effort("adaptive")` or `scripts/live_benchmark.py --adaptive` for production-quality tiered effort (tiers 1–3, refine-before-accept).
- When optimizing speed, accelerate `PoseScorer` endpoint/corner scoring and refine inner loops (batched NumPy or optional Numba) before rewriting adaptive/greedy; prefer Numba over C++ bindings unless a hard sub-20 ms median budget requires native code.
- Do not enumerate all combinatorial corner pairings before scoring; evaluate and prune hypotheses greedily.
- Target close to 100% localization success on maze maps; validate with large simulator batches (e.g. `live_benchmark.py -n 1000 --seed 42 --adaptive`) or pytest slow batches.
- Use the LIDAR simulator to generate robust automated tests for localization changes.
- Prefer complete fixes over minimal hacks when fixing localization accuracy or tier behavior.
- Only create git commits or push to remote when explicitly requested.

## Learned Workspace Facts

- Python 3.12+ uv project (`autolocalize` package) for automatic initial robot localization (LIDAR frame to map frame); published at https://github.com/ryanrudes/autolocalize with git remote named `GitHub` (not `origin`).
- Maps live under `maps/` as ROS occupancy grid YAML+PGM pairs (e.g. `churchsidemaze1.yaml`).
- `LidarSimulator` ray-casts against the occupancy grid; default config is 720 rays over 360° with configurable min/max range.
- Localization matches wall-corner features from map and LIDAR scan endpoints, refines candidates, and picks best pose; effort presets via `config_for_effort("standard"|"fast"|"adaptive")` — fast skips refine and limits corner pairs.
- Adaptive effort (tiers 1–3, no tier 0): tier 1 quick greedy + cascade `refine_pose_quick`; tier 2 refines top candidates when heap is ambiguous; tier 3 full greedy + multiscale refine + grid recovery; refine-before-accept always (no unrefined early exit).
- `pick_best_candidate` selects by endpoint score (0.03 tie band) then corner cost; do not require `_has_clear_winner` for all tier-2 exits (inflates tier-3 share and slows runs).
- Greedy quick search caps single-corner hypotheses via `max_map_corners_for_singles` to avoid huge enumeration.
- Connected-freespace checks (`autolocalize/map/freespace.py`) hard-reject impossible poses; hypothesis search ranks without per-ray freespace gating.
- Tests are pytest-based in `tests/` (`slow`/`stress` markers); `test_adaptive_localization.py` asserts ≥99.5% success and <20% tier-3; seed-42 regression indices 313, 576, 708, 777, 794, 891, 985.
- `scripts/visualize_localization.py` demos poses (`uv sync --extra viz`); `scripts/live_benchmark.py` Rich benchmark (`-n`, `--adaptive`, `--fast`, `uv sync --dev` for `rich`).
- Localize CPU time is dominated by repeated `PoseScorer` scoring (`score_fast`, `score_corners`) and refine grid search, not adaptive tier orchestration.
- CI `test_maze_localization_speed` uses median localize time < 1.2s to tolerate GitHub Actions runner variance.
