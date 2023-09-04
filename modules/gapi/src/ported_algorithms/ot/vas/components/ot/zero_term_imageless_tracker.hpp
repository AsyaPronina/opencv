// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2023 Intel Corporation

#ifndef OPENCV_GAPI_PORTED_ALGORITHMS_VAS_OT_ZERO_TERM_IMAGELESS_TRACKER_HPP
#define OPENCV_GAPI_PORTED_ALGORITHMS_VAS_OT_ZERO_TERM_IMAGELESS_TRACKER_HPP

#include <deque>
#include <vector>

#include "ported_algorithms/ot/vas/components/ot/tracker.hpp"

namespace vas {
namespace ot {

class ZeroTermImagelessTracker : public Tracker {
  public:
    explicit ZeroTermImagelessTracker(vas::ot::Tracker::InitParameters init_param);
    virtual ~ZeroTermImagelessTracker();

    virtual int32_t TrackObjects(const cv::Mat &mat, const std::vector<Detection> &detections,
                                 std::vector<std::shared_ptr<Tracklet>> *tracklets, float delta_t);

    ZeroTermImagelessTracker() = delete;
    ZeroTermImagelessTracker(const ZeroTermImagelessTracker &) = delete;
    ZeroTermImagelessTracker &operator=(const ZeroTermImagelessTracker &) = delete;

  private:
    void TrimTrajectories();
};

}; // namespace ot
}; // namespace vas

#endif // OPENCV_GAPI_PORTED_ALGORITHMS_VAS_OT_ZERO_TERM_IMAGELESS_TRACKER_HPP
