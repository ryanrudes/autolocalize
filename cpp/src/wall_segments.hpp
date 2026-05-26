#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace autolocalize {

struct Segment2D {
  double ax;
  double ay;
  double bx;
  double by;
};

struct PointSegmentDistance {
  double distance;
  double grad_x;
  double grad_y;
  bool valid;
};

PointSegmentDistance point_segment_distance(double px, double py,
                                            const Segment2D& segment);

class WallSegmentIndex {
 public:
  WallSegmentIndex() = default;

  WallSegmentIndex(const std::vector<uint8_t>& occupied, int width, int height,
                   double origin_x, double origin_y, double resolution,
                   double bucket_size);

  const std::vector<Segment2D>& segments() const { return segments_; }

  PointSegmentDistance nearest_segment(double px, double py,
                                       double max_distance) const;

 private:
  std::vector<Segment2D> segments_;
  double bucket_size_;
  int grid_width_;
  int grid_height_;
  std::vector<std::vector<int>> buckets_;

  void add_segment(double ax, double ay, double bx, double by);
  void register_segment(int segment_index);
  void bucket_coords(double px, double py, int* ix, int* iy) const;
};

std::vector<Segment2D> extract_wall_segments(const std::vector<uint8_t>& occupied,
                                             int width, int height,
                                             double origin_x, double origin_y,
                                             double resolution);

}  // namespace autolocalize
