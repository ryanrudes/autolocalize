#pragma once

#include "wall_segments.hpp"

namespace autolocalize {

struct IcpRefineResult {
  double x;
  double y;
  double theta;
  double mean_residual;
  int iterations;
  bool converged;
};

IcpRefineResult refine_icp(const WallSegmentIndex& walls,
                           const std::vector<double>& local_xy, double x, double y,
                           double theta, int max_iterations,
                           double max_association_dist,
                           double convergence_translation,
                           double convergence_rotation, double huber_delta,
                           int min_points);

}  // namespace autolocalize
