#include "refine.hpp"

#include <algorithm>
#include <cmath>

namespace autolocalize {

std::vector<double> frange(double start, double stop, double step) {
  std::vector<double> values;
  if (step <= 0.0) {
    values.push_back(start);
    return values;
  }
  for (double v = start; v <= stop + step * 0.5; v += step) {
    values.push_back(v);
  }
  return values;
}

RefineResult search_grid(const PoseScorerCore& scorer, double x, double y,
                         double theta, double translation_step,
                         double translation_span, double rotation_step,
                         double rotation_span) {
  RefineResult best{x, y, theta, scorer.score_endpoints(x, y, theta)};

  const auto dx_vals = frange(-translation_span, translation_span, translation_step);
  const auto dy_vals = frange(-translation_span, translation_span, translation_step);
  const auto dtheta_vals =
      frange(-rotation_span, rotation_span, rotation_step);

  for (double dx : dx_vals) {
    for (double dy : dy_vals) {
      for (double dtheta : dtheta_vals) {
        if (dx == 0.0 && dy == 0.0 && dtheta == 0.0) {
          continue;
        }
        const double cx = x + dx;
        const double cy = y + dy;
        const double ctheta = normalize_angle(theta + dtheta);
        const double score = scorer.score_endpoints(cx, cy, ctheta);
        if (score > best.score) {
          best = {cx, cy, ctheta, score};
        }
      }
    }
  }

  return best;
}

RefineResult refine_multiscale(const PoseScorerCore& scorer, double x, double y,
                               double theta, double translation_span,
                               double rotation_span) {
  RefineResult best{x, y, theta, scorer.score_endpoints(x, y, theta)};

  struct Scale {
    double t_span;
    double t_step;
    double r_span;
    double r_step;
  };

  const Scale scales[] = {
      {std::max(translation_span, 0.35), 0.12, std::max(rotation_span, 0.45),
       0.12},
      {0.15, 0.06, 0.25, 0.06},
      {0.06, 0.03, 0.12, 0.04},
  };

  for (const Scale& scale : scales) {
    const RefineResult candidate = search_grid(
        scorer, best.x, best.y, best.theta, scale.t_step, scale.t_span,
        scale.r_step, scale.r_span);
    if (candidate.score > best.score) {
      best = candidate;
    }
  }

  return best;
}

}  // namespace autolocalize
