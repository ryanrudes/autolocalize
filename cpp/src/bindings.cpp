#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "pose_scorer.hpp"
#include "refine.hpp"

namespace py = pybind11;

namespace {

std::vector<double> flatten_xy2(const py::array_t<double>& xy) {
  if (xy.ndim() == 1 && xy.shape(0) == 0) {
    return {};
  }
  if (xy.ndim() != 2 || xy.shape(1) != 2) {
    throw std::invalid_argument("expected Nx2 float array");
  }
  const py::ssize_t n = xy.shape(0);
  std::vector<double> out(static_cast<size_t>(2 * n));
  const double* src = xy.data();
  for (py::ssize_t i = 0; i < n; ++i) {
    out[static_cast<size_t>(2 * i)] = src[2 * i];
    out[static_cast<size_t>(2 * i + 1)] = src[2 * i + 1];
  }
  return out;
}

std::vector<double> flatten_angles(const py::array_t<double>& angles) {
  if (angles.ndim() == 1 && angles.shape(0) == 0) {
    return {};
  }
  if (angles.ndim() != 1) {
    throw std::invalid_argument("expected 1D float array");
  }
  const py::ssize_t n = angles.shape(0);
  return std::vector<double>(angles.data(), angles.data() + n);
}

std::vector<uint8_t> flatten_bool_mask(const py::array_t<bool>& mask) {
  if (mask.ndim() != 2) {
    throw std::invalid_argument("expected 2D bool mask");
  }
  const py::ssize_t h = mask.shape(0);
  const py::ssize_t w = mask.shape(1);
  std::vector<uint8_t> out(static_cast<size_t>(h * w));
  const bool* src = mask.data();
  for (py::ssize_t i = 0; i < h * w; ++i) {
    out[static_cast<size_t>(i)] = src[i] ? 1 : 0;
  }
  return out;
}


class PoseScorerNative {
 public:
  PoseScorerNative(
      py::array_t<bool> hit_mask,
      py::array_t<double> local_xy,
      py::array_t<double> scan_xy,
      py::array_t<double> scan_angles,
      py::array_t<double> map_xy,
      py::array_t<double> map_angles,
      double origin_x,
      double origin_y,
      double resolution,
      double pos_tol_sq,
      double angle_tolerance,
      bool corner_match_requires_angle,
      bool freespace_consistency,
      bool reject_robot_outside_free,
      py::object component_labels,
      std::vector<py::array_t<bool>> reachable_masks)
      : height_(static_cast<int>(hit_mask.shape(0))),
        width_(static_cast<int>(hit_mask.shape(1))) {
    if (hit_mask.ndim() != 2) {
      throw std::invalid_argument("hit_mask must be 2D");
    }

    std::vector<int32_t> labels;
    std::vector<std::vector<uint8_t>> reachable;
    if (freespace_consistency && !component_labels.is_none()) {
      auto labels_arr =
          py::array_t<int32_t, py::array::c_style | py::array::forcecast>(
              component_labels);
      if (!labels_arr || labels_arr.ndim() != 2 ||
          labels_arr.shape(0) != height_ || labels_arr.shape(1) != width_) {
        throw std::invalid_argument(
            "component_labels must match hit_mask shape");
      }
      const py::ssize_t n = height_ * width_;
      labels.assign(labels_arr.data(), labels_arr.data() + n);
      reachable.reserve(reachable_masks.size());
      for (const auto& mask : reachable_masks) {
        reachable.push_back(flatten_bool_mask(mask));
      }
    }

    core_.emplace(
        flatten_bool_mask(hit_mask),
        width_,
        height_,
        flatten_xy2(local_xy),
        flatten_xy2(scan_xy),
        flatten_angles(scan_angles),
        flatten_xy2(map_xy),
        flatten_angles(map_angles),
        origin_x,
        origin_y,
        resolution,
        pos_tol_sq,
        angle_tolerance,
        corner_match_requires_angle,
        freespace_consistency,
        reject_robot_outside_free,
        std::move(labels),
        std::move(reachable));
  }

  double score_fast(double x, double y, double theta) const {
    return core_->score_endpoints(x, y, theta);
  }

  double score_corners(double x, double y, double theta) const {
    return core_->score_corners(x, y, theta);
  }

  double corner_assignment_cost(double x, double y, double theta) const {
    return core_->corner_assignment_cost(x, y, theta);
  }

  double rank_pose(double x, double y, double theta, double corner_weight,
                   double min_ep_for_corners) const {
    return core_->rank_pose(x, y, theta, corner_weight, min_ep_for_corners);
  }

  double freespace_violation_rate(double x, double y, double theta) const {
    return core_->freespace_violation_rate(x, y, theta);
  }

  py::tuple refine_grid(double x, double y, double theta, double translation_step,
                        double translation_span, double rotation_step,
                        double rotation_span) const {
    const autolocalize::RefineResult result = autolocalize::search_grid(
        *core_, x, y, theta, translation_step, translation_span, rotation_step,
        rotation_span);
    return py::make_tuple(result.x, result.y, result.theta, result.score);
  }

  py::tuple refine_quick(double x, double y, double theta) const {
    return refine_grid(x, y, theta, 0.1, 0.16, 0.1, 0.28);
  }

  py::tuple refine(double x, double y, double theta, double translation_span,
                   double rotation_span) const {
    return refine_grid(x, y, theta, 0.08, translation_span, 0.08,
                       rotation_span);
  }

  py::tuple refine_multiscale(double x, double y, double theta,
                              double translation_span,
                              double rotation_span) const {
    const autolocalize::RefineResult result = autolocalize::refine_multiscale(
        *core_, x, y, theta, translation_span, rotation_span);
    return py::make_tuple(result.x, result.y, result.theta, result.score);
  }

 private:
  int height_;
  int width_;
  std::optional<autolocalize::PoseScorerCore> core_;
};

}  // namespace

PYBIND11_MODULE(_native, m) {
  m.doc() = "Native C++ acceleration for autolocalize pose scoring";

  py::class_<PoseScorerNative>(m, "PoseScorerNative")
      .def(py::init<py::array_t<bool>, py::array_t<double>, py::array_t<double>,
                    py::array_t<double>, py::array_t<double>, py::array_t<double>,
                    double, double, double, double, double, bool, bool, bool,
                    py::object, std::vector<py::array_t<bool>>>(),
           py::arg("hit_mask"),
           py::arg("local_xy"),
           py::arg("scan_xy"),
           py::arg("scan_angles"),
           py::arg("map_xy"),
           py::arg("map_angles"),
           py::arg("origin_x"),
           py::arg("origin_y"),
           py::arg("resolution"),
           py::arg("pos_tol_sq"),
           py::arg("angle_tolerance"),
           py::arg("corner_match_requires_angle") = false,
           py::arg("freespace_consistency") = false,
           py::arg("reject_robot_outside_free") = true,
           py::arg("component_labels") = py::none(),
           py::arg("reachable_masks") = std::vector<py::array_t<bool>>{})
      .def("score_fast", &PoseScorerNative::score_fast, py::arg("x"),
           py::arg("y"), py::arg("theta"))
      .def("score_corners", &PoseScorerNative::score_corners, py::arg("x"),
           py::arg("y"), py::arg("theta"))
      .def("corner_assignment_cost", &PoseScorerNative::corner_assignment_cost,
           py::arg("x"), py::arg("y"), py::arg("theta"))
      .def("rank_pose", &PoseScorerNative::rank_pose, py::arg("x"), py::arg("y"),
           py::arg("theta"), py::arg("corner_weight"),
           py::arg("min_ep_for_corners"))
      .def("freespace_violation_rate", &PoseScorerNative::freespace_violation_rate,
           py::arg("x"), py::arg("y"), py::arg("theta"))
      .def("refine_grid", &PoseScorerNative::refine_grid, py::arg("x"),
           py::arg("y"), py::arg("theta"), py::arg("translation_step"),
           py::arg("translation_span"), py::arg("rotation_step"),
           py::arg("rotation_span"))
      .def("refine_quick", &PoseScorerNative::refine_quick, py::arg("x"),
           py::arg("y"), py::arg("theta"))
      .def("refine", &PoseScorerNative::refine, py::arg("x"), py::arg("y"),
           py::arg("theta"), py::arg("translation_span"),
           py::arg("rotation_span"))
      .def("refine_multiscale", &PoseScorerNative::refine_multiscale, py::arg("x"),
           py::arg("y"), py::arg("theta"), py::arg("translation_span"),
           py::arg("rotation_span"));
}
