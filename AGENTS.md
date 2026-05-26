## Learned User Preferences

- Prefer feature-based initial localization over free-space grid search when accuracy allows; precompute map features for reuse.
- Expect initial localization to run in milliseconds by scoring high-value feature correspondences first and stopping early once the match score is good enough.
- Do not enumerate all combinatorial corner pairings before scoring; evaluate and prune hypotheses greedily.
- Use the LIDAR simulator to generate robust automated tests for localization changes.

## Learned Workspace Facts

- Python 3.12+ uv project (`autolocalize` package) for automatic initial robot localization (LIDAR frame to map frame).
- Maps live under `maps/` as ROS occupancy grid YAML+PGM pairs (e.g. `churchsidemaze1.yaml`).
- `LidarSimulator` ray-casts against the occupancy grid; default config is 720 rays over 360° with configurable min/max range.
- Localization matches wall-corner features extracted from the map and from LIDAR scan endpoints, then refines the best pose.
- Tests are pytest-based in `tests/`; `scripts/visualize_localization.py` demos random poses (`uv sync --extra viz` for matplotlib).
