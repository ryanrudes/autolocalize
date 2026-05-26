#include "icp_refine.hpp"

#include "pose_scorer.hpp"

#include <cmath>
#include <limits>
#include <vector>

namespace autolocalize {

namespace {

inline double huber_weight(double residual, double delta) {
  const double abs_r = std::abs(residual);
  if (abs_r <= delta || delta <= 0.0) {
    return 1.0;
  }
  return delta / abs_r;
}

bool solve_normal_equations(double jtj[9], double jtr[3], double lambda,
                            double delta[3]) {
  jtj[0] += lambda;
  jtj[4] += lambda;
  jtj[8] += lambda;

  const double a00 = jtj[0];
  const double a01 = jtj[1];
  const double a02 = jtj[2];
  const double a11 = jtj[4];
  const double a12 = jtj[5];
  const double a22 = jtj[8];

  double b0 = -jtr[0];
  double b1 = -jtr[1];
  double b2 = -jtr[2];

  double m00 = a00;
  double m01 = a01;
  double m02 = a02;
  double m11 = a11;
  double m12 = a12;
  double m22 = a22;

  if (std::abs(m00) < 1e-12) {
    return false;
  }
  const double inv00 = 1.0 / m00;
  m01 *= inv00;
  m02 *= inv00;
  m11 -= m01 * m01 * m00;
  m12 -= m01 * m02 * m00;
  b1 -= m01 * b0;
  b0 *= inv00;

  if (std::abs(m11) < 1e-12) {
    return false;
  }
  const double inv11 = 1.0 / m11;
  m12 *= inv11;
  m22 -= m12 * m12 * m11;
  b2 -= m12 * b1;
  b1 *= inv11;

  if (std::abs(m22) < 1e-12) {
    return false;
  }
  const double inv22 = 1.0 / m22;
  b2 *= inv22;
  b1 -= m12 * b2;
  b0 -= m01 * b1 + m02 * b2;

  delta[0] = b0;
  delta[1] = b1;
  delta[2] = b2;
  return true;
}

}  // namespace

IcpRefineResult refine_icp(const WallSegmentIndex& walls,
                           const std::vector<double>& local_xy, double x, double y,
                           double theta, int max_iterations,
                           double max_association_dist,
                           double convergence_translation,
                           double convergence_rotation, double huber_delta,
                           int min_points) {
  IcpRefineResult result{x, y, theta, 0.0, 0, false};
  const int num_points = static_cast<int>(local_xy.size() / 2);
  if (num_points < min_points || walls.segments().empty()) {
    return result;
  }

  double lambda = 1e-3;
  double last_cost = 1e300;

  for (int iter = 0; iter < max_iterations; ++iter) {
    const double c = std::cos(theta);
    const double s = std::sin(theta);

    double jtj[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double jtr[3] = {0.0, 0.0, 0.0};
    double cost = 0.0;
    int used = 0;

    for (int i = 0; i < num_points; ++i) {
      const double lx = local_xy[static_cast<size_t>(2 * i)];
      const double ly = local_xy[static_cast<size_t>(2 * i + 1)];
      const double wx = x + c * lx - s * ly;
      const double wy = y + s * lx + c * ly;

      const PointSegmentDistance match =
          walls.nearest_segment(wx, wy, max_association_dist);
      if (!match.valid) {
        continue;
      }

      const double residual = match.distance;
      const double weight = huber_weight(residual, huber_delta);
      const double dwx_dtheta = -s * lx - c * ly;
      const double dwy_dtheta = c * lx - s * ly;

      const double jx = match.grad_x;
      const double jy = match.grad_y;
      const double j0 = jx;
      const double j1 = jy;
      const double j2 = jx * dwx_dtheta + jy * dwy_dtheta;

      jtj[0] += weight * j0 * j0;
      jtj[1] += weight * j0 * j1;
      jtj[2] += weight * j0 * j2;
      jtj[4] += weight * j1 * j1;
      jtj[5] += weight * j1 * j2;
      jtj[8] += weight * j2 * j2;

      const double wr = weight * residual;
      jtr[0] += j0 * wr;
      jtr[1] += j1 * wr;
      jtr[2] += j2 * wr;

      cost += weight * residual * residual;
      ++used;
    }

    result.iterations = iter + 1;
    result.mean_residual =
        used > 0 ? std::sqrt(cost / static_cast<double>(used)) : 0.0;

    if (used < min_points) {
      return result;
    }

    double delta[3] = {0.0, 0.0, 0.0};
    if (!solve_normal_equations(jtj, jtr, lambda, delta)) {
      return result;
    }

    const double new_x = x + delta[0];
    const double new_y = y + delta[1];
    const double new_theta = autolocalize::normalize_angle(theta + delta[2]);

    if (cost >= last_cost) {
      lambda = std::min(lambda * 10.0, 1e6);
      continue;
    }

    x = new_x;
    y = new_y;
    theta = new_theta;
    lambda = std::max(lambda * 0.3, 1e-6);
    last_cost = cost;

    if (std::hypot(delta[0], delta[1]) <= convergence_translation &&
        std::abs(delta[2]) <= convergence_rotation) {
      result.converged = true;
      break;
    }
  }

  result.x = x;
  result.y = y;
  result.theta = theta;
  return result;
}

}  // namespace autolocalize
