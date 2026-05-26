#include "pose_scorer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace autolocalize {

namespace {

inline int floor_to_int(double v) {
  return static_cast<int>(std::floor(v));
}

inline int clamp_int(int v, int lo, int hi) {
  return std::max(lo, std::min(v, hi));
}

}  // namespace

PoseScorerCore::PoseScorerCore(
    std::vector<uint8_t> hit_mask,
    int width,
    int height,
    std::vector<double> local_xy,
    std::vector<double> scan_xy,
    std::vector<double> scan_angles,
    std::vector<double> map_xy,
    std::vector<double> map_angles,
    double origin_x,
    double origin_y,
    double resolution,
    double pos_tol_sq,
    double angle_tolerance,
    bool corner_match_requires_angle,
    bool freespace_consistency,
    bool reject_robot_outside_free,
    std::vector<int32_t> component_labels,
    std::vector<std::vector<uint8_t>> reachable_masks)
    : hit_mask_(std::move(hit_mask)),
      width_(width),
      height_(height),
      local_xy_(std::move(local_xy)),
      scan_xy_(std::move(scan_xy)),
      scan_angles_(std::move(scan_angles)),
      map_xy_(std::move(map_xy)),
      map_angles_(std::move(map_angles)),
      origin_x_(origin_x),
      origin_y_(origin_y),
      resolution_(resolution),
      pos_tol_sq_(pos_tol_sq),
      angle_tolerance_(angle_tolerance),
      corner_match_requires_angle_(corner_match_requires_angle),
      freespace_consistency_(freespace_consistency),
      reject_robot_outside_free_(reject_robot_outside_free),
      component_labels_(std::move(component_labels)),
      reachable_masks_(std::move(reachable_masks)) {}

int PoseScorerCore::robot_component(int gx, int gy) const {
  if (component_labels_.empty()) {
    return -1;
  }
  if (gx < 0 || gx >= width_ || gy < 0 || gy >= height_) {
    return -1;
  }
  const int label = component_labels_[static_cast<size_t>(gy * width_ + gx)];
  return label >= 0 ? label : -1;
}

bool PoseScorerCore::endpoint_plausible(int gx, int gy,
                                        const uint8_t* reachable) const {
  if (gx >= 0 && gx < width_ && gy >= 0 && gy < height_) {
    const int idx = gy * width_ + gx;
    return hit_mask_[static_cast<size_t>(idx)] != 0 || reachable[idx] != 0;
  }
  const int cx = clamp_int(gx, 0, width_ - 1);
  const int cy = clamp_int(gy, 0, height_ - 1);
  return reachable[static_cast<size_t>(cy * width_ + cx)] != 0;
}

bool PoseScorerCore::score_endpoints_with_reachable(
    double x, double y, double theta, const uint8_t* reachable,
    int* hits_out) const {
  const int n = num_rays();
  if (n == 0) {
    *hits_out = 0;
    return true;
  }

  const double c = std::cos(theta);
  const double s = std::sin(theta);
  int hits = 0;

  for (int i = 0; i < n; ++i) {
    const double lx = local_xy_[static_cast<size_t>(2 * i)];
    const double ly = local_xy_[static_cast<size_t>(2 * i + 1)];
    const double wx = x + c * lx - s * ly;
    const double wy = y + s * lx + c * ly;
    const int gx = floor_to_int((wx - origin_x_) / resolution_);
    const int gy = floor_to_int((wy - origin_y_) / resolution_);

    if (freespace_consistency_ && reachable != nullptr) {
      if (!endpoint_plausible(gx, gy, reachable)) {
        *hits_out = 0;
        return false;
      }
    }

    if (gx >= 0 && gx < width_ && gy >= 0 && gy < height_) {
      if (hit_mask_[static_cast<size_t>(gy * width_ + gx)] != 0) {
        ++hits;
      }
    }
  }

  *hits_out = hits;
  return true;
}

double PoseScorerCore::score_endpoints(double x, double y, double theta) const {
  const int n = num_rays();
  if (n == 0) {
    return 0.0;
  }

  const uint8_t* reachable = nullptr;
  if (freespace_consistency_ && !component_labels_.empty()) {
    const int robot_gx = floor_to_int((x - origin_x_) / resolution_);
    const int robot_gy = floor_to_int((y - origin_y_) / resolution_);
    const int component = robot_component(robot_gx, robot_gy);
    if (component < 0) {
      if (reject_robot_outside_free_) {
        return 0.0;
      }
    } else if (component >= static_cast<int>(reachable_masks_.size())) {
      if (reject_robot_outside_free_) {
        return 0.0;
      }
    } else {
      reachable = reachable_masks_[static_cast<size_t>(component)].data();
    }
  }

  int hits = 0;
  if (!score_endpoints_with_reachable(x, y, theta, reachable, &hits)) {
    return 0.0;
  }
  return static_cast<double>(hits) / static_cast<double>(n);
}

double PoseScorerCore::score_corners(double x, double y, double theta) const {
  const int n_scan = num_scan();
  const int n_map = num_map();
  if (n_scan == 0 || n_map == 0) {
    return 0.0;
  }

  const double c = std::cos(theta);
  const double s = std::sin(theta);
  int matched = 0;

  for (int i = 0; i < n_scan; ++i) {
    const double lx = scan_xy_[static_cast<size_t>(2 * i)];
    const double ly = scan_xy_[static_cast<size_t>(2 * i + 1)];
    const double wx = x + c * lx - s * ly;
    const double wy = y + s * lx + c * ly;
    const double world_angle =
        normalize_angle(theta + scan_angles_[static_cast<size_t>(i)]);

    bool any = false;
    for (int j = 0; j < n_map; ++j) {
      const double dx = wx - map_xy_[static_cast<size_t>(2 * j)];
      const double dy = wy - map_xy_[static_cast<size_t>(2 * j + 1)];
      const double dist_sq = dx * dx + dy * dy;
      if (dist_sq > pos_tol_sq_) {
        continue;
      }
      if (corner_match_requires_angle_) {
        const double angle_diff = std::abs(
            normalize_angle(world_angle -
                            map_angles_[static_cast<size_t>(j)]));
        if (angle_diff > angle_tolerance_) {
          continue;
        }
      }
      any = true;
      break;
    }
    if (any) {
      ++matched;
    }
  }

  return static_cast<double>(matched) / static_cast<double>(n_scan);
}

double PoseScorerCore::corner_assignment_cost(double x, double y,
                                              double theta) const {
  const int n_scan = num_scan();
  const int n_map = num_map();
  if (n_scan == 0 || n_map == 0) {
    return 1e100;
  }

  const double c = std::cos(theta);
  const double s = std::sin(theta);

  std::vector<double> min_dist(static_cast<size_t>(n_scan));
  std::vector<int> order(static_cast<size_t>(n_scan));
  for (int i = 0; i < n_scan; ++i) {
    order[static_cast<size_t>(i)] = i;
    const double lx = scan_xy_[static_cast<size_t>(2 * i)];
    const double ly = scan_xy_[static_cast<size_t>(2 * i + 1)];
    const double wx = x + c * lx - s * ly;
    const double wy = y + s * lx + c * ly;
    double best = 1e100;
    for (int j = 0; j < n_map; ++j) {
      const double dx = wx - map_xy_[static_cast<size_t>(2 * j)];
      const double dy = wy - map_xy_[static_cast<size_t>(2 * j + 1)];
      const double d = std::sqrt(dx * dx + dy * dy);
      if (d < best) {
        best = d;
      }
    }
    min_dist[static_cast<size_t>(i)] = best;
  }

  std::sort(order.begin(), order.end(),
            [&](int a, int b) { return min_dist[a] < min_dist[b]; });

  std::vector<bool> used_map(static_cast<size_t>(n_map), false);
  double total = 0.0;

  for (int scan_idx : order) {
    const double lx = scan_xy_[static_cast<size_t>(2 * scan_idx)];
    const double ly = scan_xy_[static_cast<size_t>(2 * scan_idx + 1)];
    const double wx = x + c * lx - s * ly;
    const double wy = y + s * lx + c * ly;

    int best_j = -1;
    double best_d = 1e100;
    for (int j = 0; j < n_map; ++j) {
      if (used_map[static_cast<size_t>(j)]) {
        continue;
      }
      const double dx = wx - map_xy_[static_cast<size_t>(2 * j)];
      const double dy = wy - map_xy_[static_cast<size_t>(2 * j + 1)];
      const double d = std::sqrt(dx * dx + dy * dy);
      if (d < best_d) {
        best_d = d;
        best_j = j;
      }
    }
    if (best_j < 0) {
      return 1e100;
    }
    used_map[static_cast<size_t>(best_j)] = true;
    total += best_d;
  }

  return total / static_cast<double>(n_scan);
}

double PoseScorerCore::rank_pose(double x, double y, double theta,
                                 double corner_weight,
                                 double min_ep_for_corners) const {
  const double endpoint = score_endpoints(x, y, theta);
  if (endpoint < min_ep_for_corners) {
    return endpoint;
  }
  if (corner_weight > 0.0 && num_scan() > 0) {
    return endpoint + corner_weight * score_corners(x, y, theta);
  }
  return endpoint;
}

double PoseScorerCore::freespace_violation_rate(double x, double y,
                                                double theta) const {
  const int n = num_rays();
  if (n == 0 || component_labels_.empty()) {
    return 1.0;
  }

  const int robot_gx = floor_to_int((x - origin_x_) / resolution_);
  const int robot_gy = floor_to_int((y - origin_y_) / resolution_);
  const int component = robot_component(robot_gx, robot_gy);
  if (component < 0 ||
      component >= static_cast<int>(reachable_masks_.size())) {
    return 1.0;
  }

  const uint8_t* reachable =
      reachable_masks_[static_cast<size_t>(component)].data();
  const double c = std::cos(theta);
  const double s = std::sin(theta);
  int violations = 0;

  for (int i = 0; i < n; ++i) {
    const double lx = local_xy_[static_cast<size_t>(2 * i)];
    const double ly = local_xy_[static_cast<size_t>(2 * i + 1)];
    const double wx = x + c * lx - s * ly;
    const double wy = y + s * lx + c * ly;
    const int gx = floor_to_int((wx - origin_x_) / resolution_);
    const int gy = floor_to_int((wy - origin_y_) / resolution_);
    if (!endpoint_plausible(gx, gy, reachable)) {
      ++violations;
    }
  }

  return static_cast<double>(violations) / static_cast<double>(n);
}

}  // namespace autolocalize
