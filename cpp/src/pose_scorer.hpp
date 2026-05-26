#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

namespace autolocalize {

inline double normalize_angle(double angle) {
  return std::remainder(angle, 2.0 * M_PI);
}

class PoseScorerCore {
 public:
  PoseScorerCore(
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
      std::vector<std::vector<uint8_t>> reachable_masks);

  double score_endpoints(double x, double y, double theta) const;
  double score_corners(double x, double y, double theta) const;
  double corner_assignment_cost(double x, double y, double theta) const;
  double rank_pose(double x, double y, double theta, double corner_weight,
                   double min_ep_for_corners) const;
  double freespace_violation_rate(double x, double y, double theta) const;

 private:
  std::vector<uint8_t> hit_mask_;
  int width_;
  int height_;
  std::vector<double> local_xy_;
  std::vector<double> scan_xy_;
  std::vector<double> scan_angles_;
  std::vector<double> map_xy_;
  std::vector<double> map_angles_;
  double origin_x_;
  double origin_y_;
  double resolution_;
  double pos_tol_sq_;
  double angle_tolerance_;
  bool corner_match_requires_angle_;
  bool freespace_consistency_;
  bool reject_robot_outside_free_;
  std::vector<int32_t> component_labels_;
  std::vector<std::vector<uint8_t>> reachable_masks_;

  int num_rays() const { return static_cast<int>(local_xy_.size() / 2); }
  int num_scan() const { return static_cast<int>(scan_xy_.size() / 2); }
  int num_map() const { return static_cast<int>(map_xy_.size() / 2); }

  int robot_component(int gx, int gy) const;
  bool endpoint_plausible(int gx, int gy, const uint8_t* reachable) const;
  bool score_endpoints_with_reachable(double x, double y, double theta,
                                      const uint8_t* reachable,
                                      int* hits_out) const;
};

}  // namespace autolocalize
