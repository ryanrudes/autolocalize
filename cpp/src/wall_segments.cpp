#include "wall_segments.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace autolocalize {

namespace {

inline bool is_occupied(const std::vector<uint8_t>& occupied, int width, int height,
                        int gx, int gy) {
  if (gx < 0 || gx >= width || gy < 0 || gy >= height) {
    return false;
  }
  return occupied[static_cast<size_t>(gy * width + gx)] != 0;
}

}  // namespace

PointSegmentDistance point_segment_distance(double px, double py,
                                            const Segment2D& segment) {
  const double vx = segment.bx - segment.ax;
  const double vy = segment.by - segment.ay;
  const double wx = px - segment.ax;
  const double wy = py - segment.ay;
  const double len_sq = vx * vx + vy * vy;

  PointSegmentDistance result{0.0, 0.0, 0.0, false};
  if (len_sq <= 1e-12) {
    const double dx = px - segment.ax;
    const double dy = py - segment.ay;
    const double dist = std::hypot(dx, dy);
    if (dist <= 1e-12) {
      return result;
    }
    result.distance = dist;
    result.grad_x = dx / dist;
    result.grad_y = dy / dist;
    result.valid = true;
    return result;
  }

  double t = (wx * vx + wy * vy) / len_sq;
  t = std::max(0.0, std::min(1.0, t));
  const double cx = segment.ax + t * vx;
  const double cy = segment.ay + t * vy;
  const double dx = px - cx;
  const double dy = py - cy;
  const double dist = std::hypot(dx, dy);
  if (dist <= 1e-12) {
    return result;
  }
  result.distance = dist;
  result.grad_x = dx / dist;
  result.grad_y = dy / dist;
  result.valid = true;
  return result;
}

std::vector<Segment2D> extract_wall_segments(const std::vector<uint8_t>& occupied,
                                             int width, int height,
                                             double origin_x, double origin_y,
                                             double resolution) {
  std::vector<Segment2D> segments;
  segments.reserve(static_cast<size_t>(width * height));

  for (int gy = 0; gy < height; ++gy) {
    for (int gx = 0; gx < width; ++gx) {
      if (!is_occupied(occupied, width, height, gx, gy)) {
        continue;
      }

      const double x0 = origin_x + static_cast<double>(gx) * resolution;
      const double y0 = origin_y + static_cast<double>(gy) * resolution;
      const double x1 = x0 + resolution;
      const double y1 = y0 + resolution;

      if (!is_occupied(occupied, width, height, gx, gy + 1)) {
        segments.push_back({x0, y1, x1, y1});
      }
      if (!is_occupied(occupied, width, height, gx, gy - 1)) {
        segments.push_back({x0, y0, x1, y0});
      }
      if (!is_occupied(occupied, width, height, gx - 1, gy)) {
        segments.push_back({x0, y0, x0, y1});
      }
      if (!is_occupied(occupied, width, height, gx + 1, gy)) {
        segments.push_back({x1, y0, x1, y1});
      }
    }
  }

  return segments;
}

WallSegmentIndex::WallSegmentIndex(const std::vector<uint8_t>& occupied, int width,
                                   int height, double origin_x, double origin_y,
                                   double resolution, double bucket_size)
    : segments_(extract_wall_segments(occupied, width, height, origin_x, origin_y,
                                      resolution)),
      bucket_size_(std::max(0.25, bucket_size)) {
  if (segments_.empty()) {
    return;
  }

  double min_x = segments_[0].ax;
  double min_y = segments_[0].ay;
  double max_x = min_x;
  double max_y = min_y;
  for (const Segment2D& segment : segments_) {
    min_x = std::min({min_x, segment.ax, segment.bx});
    min_y = std::min({min_y, segment.ay, segment.by});
    max_x = std::max({max_x, segment.ax, segment.bx});
    max_y = std::max({max_y, segment.ay, segment.by});
  }

  grid_width_ = std::max(
      1, static_cast<int>(std::ceil((max_x - min_x) / bucket_size_)) + 2);
  grid_height_ = std::max(
      1, static_cast<int>(std::ceil((max_y - min_y) / bucket_size_)) + 2);
  buckets_.assign(static_cast<size_t>(grid_width_ * grid_height_), {});

  for (int i = 0; i < static_cast<int>(segments_.size()); ++i) {
    register_segment(i);
  }
}

void WallSegmentIndex::add_segment(double ax, double ay, double bx, double by) {
  segments_.push_back({ax, ay, bx, by});
}

void WallSegmentIndex::bucket_coords(double px, double py, int* ix,
                                     int* iy) const {
  *ix = static_cast<int>(std::floor(px / bucket_size_));
  *iy = static_cast<int>(std::floor(py / bucket_size_));
}

void WallSegmentIndex::register_segment(int segment_index) {
  const Segment2D& segment = segments_[static_cast<size_t>(segment_index)];
  int min_ix = 0;
  int min_iy = 0;
  int max_ix = 0;
  int max_iy = 0;
  bucket_coords(std::min(segment.ax, segment.bx), std::min(segment.ay, segment.by),
                &min_ix, &min_iy);
  bucket_coords(std::max(segment.ax, segment.bx), std::max(segment.ay, segment.by),
                &max_ix, &max_iy);

  for (int iy = min_iy; iy <= max_iy; ++iy) {
    for (int ix = min_ix; ix <= max_ix; ++ix) {
      const int wrapped_x = ((ix % grid_width_) + grid_width_) % grid_width_;
      const int wrapped_y = ((iy % grid_height_) + grid_height_) % grid_height_;
      buckets_[static_cast<size_t>(wrapped_y * grid_width_ + wrapped_x)].push_back(
          segment_index);
    }
  }
}

PointSegmentDistance WallSegmentIndex::nearest_segment(double px, double py,
                                                       double max_distance) const {
  PointSegmentDistance best{max_distance, 0.0, 0.0, false};
  if (segments_.empty()) {
    return best;
  }

  int ix = 0;
  int iy = 0;
  bucket_coords(px, py, &ix, &iy);
  const int radius =
      std::max(1, static_cast<int>(std::ceil(max_distance / bucket_size_)));

  for (int dy = -radius; dy <= radius; ++dy) {
    for (int dx = -radius; dx <= radius; ++dx) {
      const int bucket_x = ((ix + dx) % grid_width_ + grid_width_) % grid_width_;
      const int bucket_y =
          ((iy + dy) % grid_height_ + grid_height_) % grid_height_;
      const auto& bucket =
          buckets_[static_cast<size_t>(bucket_y * grid_width_ + bucket_x)];
      for (int segment_index : bucket) {
        const PointSegmentDistance candidate = point_segment_distance(
            px, py, segments_[static_cast<size_t>(segment_index)]);
        if (candidate.valid && candidate.distance < best.distance) {
          best = candidate;
        }
      }
    }
  }

  best.valid = best.valid && best.distance <= max_distance;
  return best;
}

}  // namespace autolocalize
