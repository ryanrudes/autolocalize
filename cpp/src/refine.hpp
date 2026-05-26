#pragma once

#include <vector>

#include "pose_scorer.hpp"

namespace autolocalize {

struct RefineResult {
  double x;
  double y;
  double theta;
  double score;
};

std::vector<double> frange(double start, double stop, double step);

RefineResult search_grid(const PoseScorerCore& scorer, double x, double y,
                         double theta, double translation_step,
                         double translation_span, double rotation_step,
                         double rotation_span);

RefineResult refine_multiscale(const PoseScorerCore& scorer, double x, double y,
                               double theta, double translation_span,
                               double rotation_span);

}  // namespace autolocalize
