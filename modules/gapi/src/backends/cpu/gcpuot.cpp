// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2023 Intel Corporation

#include <opencv2/gapi/ot.hpp>
#include <opencv2/gapi/cpu/ot.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>

#include <vas/ot.hpp>

namespace cv
{
namespace gapi
{
namespace ot
{
using TrackedInfoTuple = std::tuple<std::vector<cv::Rect>, std::vector<uint64_t>, std::vector<uint64_t>>;
struct TrackedObjectInfo
{
    cv::Rect object;            /* Tracked rectangle. */
    uint64_t tracking_id;       /* Tracking ID. Numbering sequence starts from 1. */
    int32_t class_label;        /* Class label. This is specified by the detected_class_labels
                                   which are used in cv::gapi::ot::Track */
    int32_t status;             /* Tracking status. */
    int32_t association_index;  /* Index in the detected_rects which are used in
                                   cv::gapi::ot::Track. */
};

struct TrackedObjects {
    using object_id_t = uint64_t;
    using counter_t = uint64_t;

    // State: hash map from object id to it's counter through the video stream (until it's been updated)
    std::unordered_map<object_id_t,counter_t> m_map;
    std::vector<cv::Rect> m_objs;
    std::vector<uint64_t> m_tr_ids;
    std::vector<uint64_t> m_lost_ids;
    size_t m_interval;
    std::shared_ptr<vas::ot::ObjectTracker> m_tracker;

    TrackedObjects(std::shared_ptr<vas::ot::ObjectTracker>&& tracker,
                   size_t interval);
    void update(cv::gapi::ot::TrackedObjectInfo&& tracked_object);
    void process_lost_id(object_id_t id);
    void process_tracked_id(object_id_t id, const cv::Rect& obj);
    void process_new_id(object_id_t id, const cv::Rect& obj);
    void cleanup();
    TrackedInfoTuple get_info();
};

TrackedObjects::TrackedObjects(std::shared_ptr<vas::ot::ObjectTracker>&& tracker,
                               size_t interval): m_interval(interval), m_tracker(tracker) {}

void TrackedObjects::cleanup() {
    m_map.clear();
    m_objs.clear();
    m_tr_ids.clear();
    m_lost_ids.clear();
}

TrackedInfoTuple TrackedObjects::get_info() {
    return std::make_tuple(m_objs, m_tr_ids, m_lost_ids);
}

void TrackedObjects::process_lost_id(object_id_t id) {
    auto it = m_map.find(id);
    if (it != m_map.end()) {
        // Stop tracking object
        m_lost_ids.push_back(id);
        m_map.erase(it);
    }
}

void TrackedObjects::process_tracked_id(object_id_t id, const cv::Rect& obj) {
    auto it = m_map.find(id);
    if (it != m_map.end()) {
        auto val = it->second;
        if (val < m_interval) {
            // Continue tracking
            ++(it->second);
        } else {
            // Reset tracking and reclassify
            it->second = 0;
            m_objs.push_back(obj);
            m_tr_ids.push_back(it->first);
        }
    }
}

void TrackedObjects::process_new_id(object_id_t id, const cv::Rect& obj) {
    auto it = m_map.find(id);
    if (it != m_map.end()) {
        // Actually not new - continue tracking
        process_tracked_id(id, obj);
    } else {
        // Start tracking object
        m_map[id] = 0;
        m_objs.push_back(obj);
        m_tr_ids.push_back(id);
    }
}

void TrackedObjects::update(cv::gapi::ot::TrackedObjectInfo&& tracked_object) {
    switch (TrackingStatus(tracked_object.status)) {
        case TrackingStatus::LOST:
        {
            process_lost_id(tracked_object.tracking_id);
            break;
        }
        case TrackingStatus::NEW:
        {
            process_new_id(tracked_object.tracking_id, tracked_object.object);
            break;
        }
        case TrackingStatus::TRACKED:
        {
            process_tracked_id(tracked_object.tracking_id, tracked_object.object);
            break;
        }
        default:
            cv::util::throw_error(std::logic_error("Unsupported tracking status"));
    }
}

// Helper functions for OT kernels
namespace {
void GTrackImplSetup(cv::GArrayDesc, cv::GArrayDesc, float,
                      std::shared_ptr<TrackedObjects>& state,
                      const ObjectTrackerParams& params) {

    GAPI_Assert(params.tracking_type == 5 && "Only ZERO_TERM_IMAGLESS tracking is supported for now");

    vas::ot::ObjectTracker::Builder ot_builder;
    ot_builder.backend_type = vas::BackendType(params.backend_type);
    ot_builder.max_num_objects = params.max_num_objects;
    ot_builder.input_image_format = vas::ColorFormat(params.input_image_format);
    ot_builder.platform_config = params.platform_config;
    ot_builder.tracking_per_class = params.tracking_per_class;

    state = std::shared_ptr<TrackedObjects>(
            new TrackedObjects(ot_builder.Build(vas::ot::TrackingType(params.tracking_type)),
                                params.reclassify_interval));
}

void GTrackImplPrepare(const std::vector<cv::Rect>& in_rects,
                       const std::vector<int32_t>& in_class_labels, float delta,
                       std::vector<vas::ot::DetectedObject>& detected_objs,
                       TrackedObjects& state)
{
    if (in_rects.size() != in_class_labels.size())
    {
        cv::util::throw_error(std::invalid_argument("Track() implementation run() method: in_rects and in_class_labels "
                                                    "sizes are different."));
    }

    detected_objs.reserve(in_rects.size());

    std::size_t n_detected_objects = in_rects.size();
    for (std::size_t i = 0; i < n_detected_objects; ++i)
    {
        detected_objs.emplace_back(in_rects[i], in_class_labels[i]);
    }

    state.m_tracker->SetFrameDeltaTime(delta);
}
} // anonymous namespace

GAPI_OCV_KERNEL_ST(GTrackFromMatImpl, cv::gapi::ot::GTrackFromMat, TrackedObjects)
{
    static void setup(cv::GMatDesc, cv::GArrayDesc rects_desc,
                      cv::GArrayDesc labels_desc, float delta,
                      std::shared_ptr<TrackedObjects>& state,
                      const cv::GCompileArgs& compile_args)
    {
        auto params = cv::gapi::getCompileArg<ObjectTrackerParams>(compile_args)
            .value_or(ObjectTrackerParams{});

        GAPI_Assert(params.input_image_format == 0 && "Only BGR input as cv::GMat is supported for now");
        GTrackImplSetup(rects_desc, labels_desc, delta, state, params);
    }

    static void run(const cv::Mat& in_mat, const std::vector<cv::Rect>& in_rects,
                    const std::vector<int32_t>& in_class_labels, float delta,
                    std::vector<cv::Rect>& out_rects, std::vector<uint64_t>& out_tr_ids,
                    std::vector<uint64_t>& out_lost_ids, TrackedObjects& state)
    {
        std::vector<vas::ot::DetectedObject> detected_objs;
        GTrackImplPrepare(in_rects, in_class_labels, delta, detected_objs, state);

        GAPI_Assert(in_mat.type() == CV_8UC3 && "Input mat is not in BGR format");

        auto objects = state.m_tracker->Track(in_mat, detected_objs);

        for (auto&& object : objects)
        {
            state.update(TrackedObjectInfo{object.rect,
                                           object.tracking_id,
                                           object.class_label,
                                           static_cast<int32_t>(object.status),
                                           object.association_idx});
        }

        std::tie(out_rects, out_tr_ids, out_lost_ids) = state.get_info();
        state.cleanup();
    }
};

GAPI_OCV_KERNEL_ST(GTrackFromFrameImpl, cv::gapi::ot::GTrackFromFrame, TrackedObjects)
{
    static void setup(cv::GFrameDesc, cv::GArrayDesc rects_desc,
                      cv::GArrayDesc labels_desc, float delta,
                      std::shared_ptr<TrackedObjects>& state,
                      const cv::GCompileArgs& compile_args)
    {
        auto params = cv::gapi::getCompileArg<ObjectTrackerParams>(compile_args)
            .value_or(ObjectTrackerParams{});

        GAPI_Assert(params.input_image_format == 1 && "Only NV12 input as cv::GFrame is supported for now");
        GTrackImplSetup(rects_desc, labels_desc, delta, state, params);
    }

    static void run(const cv::MediaFrame& in_frame, const std::vector<cv::Rect>& in_rects,
                    const std::vector<int32_t>& in_class_labels, float delta,
                    std::vector<cv::Rect>& out_rects, std::vector<uint64_t>& out_tr_ids,
                    std::vector<uint64_t>& out_lost_ids, TrackedObjects& state)
    {
        std::vector<vas::ot::DetectedObject> detected_objs;
        GTrackImplPrepare(in_rects, in_class_labels, delta, detected_objs, state);

        // Extract metadata from MediaFrame and construct cv::Mat atop of it
        cv::MediaFrame::View view = in_frame.access(cv::MediaFrame::Access::R);
        auto ptrs = view.ptr;
        auto strides = view.stride;
        auto desc = in_frame.desc();

        GAPI_Assert((desc.fmt == cv::MediaFormat::NV12 || desc.fmt == cv::MediaFormat::BGR) \
                    && "Input frame is not in NV12 or BGR format");

        cv::Mat in;
        if (desc.fmt == cv::MediaFormat::NV12) {
            GAPI_Assert(ptrs[0] != nullptr && "Y plane pointer is empty");
            GAPI_Assert(ptrs[1] != nullptr && "UV plane pointer is empty");
            if (strides[0] > 0) {
                in = cv::Mat(desc.size, CV_8UC1, ptrs[0], strides[0]);
            } else {
                in = cv::Mat(desc.size, CV_8UC1, ptrs[0]);
            }
        }

        auto objects = state.m_tracker->Track(in, detected_objs);

        for (auto&& object : objects)
        {
            state.update(TrackedObjectInfo{object.rect,
                                           object.tracking_id,
                                           object.class_label,
                                           static_cast<int32_t>(object.status),
                                           object.association_idx});
        }

        std::tie(out_rects, out_tr_ids, out_lost_ids) = state.get_info();
        state.cleanup();
    }
};

cv::gapi::GKernelPackage cpu::kernels()
{
    return cv::gapi::kernels
        <
          GTrackFromFrameImpl,
          GTrackFromMatImpl
        >();
}

}   // namespace ot
}   // namespace gapi
}   // namespace cv
