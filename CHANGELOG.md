# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05-26

### Added

- ROS-style occupancy grid loading (YAML + PGM)
- 2D LIDAR ray-cast simulator
- Feature-based initial localizer with greedy top-K search and multi-scale refinement
- Connected free-space consistency checks for pose scoring
- Coarse grid fallback when scan endpoint match is weak
- Pytest suite including 1000-pose maze Monte Carlo stress test
- Optional matplotlib visualization script (`scripts/visualize_localization.py`)
