// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2023 Intel Corporation


#include "../test_precomp.hpp"

#include <opencv2/gapi/ot.hpp>
#include <opencv2/gapi/cpu/ot.hpp>

#include "opencv2/gapi/streaming/meta.hpp"
#include "opencv2/gapi/streaming/cap.hpp"

namespace opencv_test {
struct FrameDetections {
    std::size_t frame_no{};
    std::vector<std::vector<cv::Rect>> boxes;
    std::vector<std::vector<int>> box_ids;
};

struct FrameLabels {
    std::size_t frame_no{};
    std::vector<std::vector<cv::Rect>> boxes;
    std::vector<std::vector<std::string>> labels;
};

struct FrameDetectionsParams {
    FrameDetections value;
};

struct FrameLabelsParams {
    FrameLabels value;
};
} // namespace opencv_test

namespace cv {
    namespace detail {
        template<> struct CompileArgTag<opencv_test::FrameDetectionsParams> {
            static const char* tag() {
                return "org.opencv.test.frame_detections_params";
            }
        };

        template<> struct CompileArgTag<opencv_test::FrameLabelsParams> {
            static const char* tag() {
                return "org.opencv.test.frame_labels_params";
            }
        };
    } // namespace detail
} // namespace cv

namespace opencv_test {
G_API_OP(CvVideo768x576_Detect, <std::tuple<cv::GArray<cv::Rect>, cv::GArray<int>>(cv::GMat)>,
    "test.custom.cv_video_768x576_detect") {
    static std::tuple<cv::GArrayDesc, cv::GArrayDesc> outMeta(cv::GMatDesc) {
        return std::make_tuple(cv::empty_array_desc(), cv::empty_array_desc());
    }
};

GAPI_OCV_KERNEL_ST(OCV_CvVideo768x576_Detect, CvVideo768x576_Detect, FrameDetections) {
    static void setup(cv::GMatDesc,
        std::shared_ptr<FrameDetections>&state,
        const cv::GCompileArgs & compileArgs) {
        auto params = cv::gapi::getCompileArg<opencv_test::FrameDetectionsParams>(compileArgs)
            .value_or(opencv_test::FrameDetectionsParams{ });
        state = std::make_shared<FrameDetections>(params.value);
    }

    static void run(const cv::Mat&,
        std::vector<cv::Rect>&out_boxes,
        std::vector<int>&out_box_ids,
        FrameDetections & state) {
        if (state.frame_no < state.boxes.size()) {
            out_boxes = state.boxes[state.frame_no];
            out_box_ids = state.box_ids[state.frame_no];
            ++state.frame_no;
        }
    }
};

G_API_OP(CvVideo768x576_Classify, <cv::GArray<std::string>(cv::GArray<cv::Rect>)>,
    "test.custom.cv_video_768x576_classify") {
    static cv::GArrayDesc outMeta(cv::GArrayDesc) {
        return cv::empty_array_desc();
    }
};

GAPI_OCV_KERNEL_ST(OCV_CvVideo768x576_Classify, CvVideo768x576_Classify,
    FrameLabels) {
    static void setup(cv::GArrayDesc,
        std::shared_ptr<FrameLabels>&state,
        const cv::GCompileArgs & compileArgs) {
        auto params = cv::gapi::getCompileArg<opencv_test::FrameLabelsParams>(compileArgs)
            .value_or(opencv_test::FrameLabelsParams{ });
        state = std::make_shared<FrameLabels>(params.value);
    }

    static void run(const std::vector<cv::Rect>&rois,
        std::vector<std::string>&out_labels,
        FrameLabels & state) {
        if (state.frame_no < state.boxes.size()) {
            out_labels.resize(rois.size());

            auto all_boxes = state.boxes[state.frame_no];
            auto all_labels = state.labels[state.frame_no];

            for (const auto& roi : rois) {
                for (std::size_t i = 0; i < all_labels.size(); ++i) {
                    if (roi == all_boxes[i]) {
                        out_labels[i] = all_labels[i];
                    }
                }
            }

            ++state.frame_no;
        }
    }
};

using LostIds = cv::GArray<uint64_t>;
using ROIs = cv::GArray<cv::Rect>;
using TrackedIds = cv::GArray<uint64_t>;

G_API_OP(TrackedFilterOutOfBounds, <std::tuple<ROIs, TrackedIds>(ROIs, TrackedIds)>,
    "test.custom.tracked_filter_out_of_bounds") {
    static std::tuple<cv::GArrayDesc, cv::GArrayDesc> outMeta(cv::GArrayDesc, cv::GArrayDesc) {
        return std::make_tuple(cv::empty_array_desc(), cv::empty_array_desc());
    }
};

GAPI_OCV_KERNEL(OCVTrackedFilterOutOfBounds, TrackedFilterOutOfBounds) {
    static void run(const std::vector<cv::Rect>&in_rcts,
        const std::vector<uint64_t>&in_tr_ids,
        std::vector<cv::Rect>&out_rcts,
        std::vector<uint64_t>&out_tr_ids) {
        static int frame_no;

        static cv::FileStorage trackings("ot_trackings.yml", cv::FileStorage::WRITE);
        trackings << "Frame_" + std::to_string(frame_no) << "{";

        for (uint32_t i = 0; i < in_rcts.size(); ++i) {
            const cv::Rect rc = in_rcts[i];

            trackings << "box_" + std::to_string(i) << "{";

            out_tr_ids.push_back(in_tr_ids[i]);
            trackings << "tracking_id" << int(out_tr_ids[i]);

            out_rcts.push_back(rc);
            trackings << "x" << rc.x << "y" << rc.y;
            trackings << "width" << rc.width << "height" << rc.height;
            trackings << "}";
        }
        trackings << "}";

        ++frame_no;
    }
};

// State: hash map from object id to it's class
struct TrackedClasses {
    std::unordered_map<uint64_t, std::string> m_map;
};

using TrackedIds = cv::GArray<uint64_t>;
using LostIds = cv::GArray<uint64_t>;

G_API_OP(TrackClasses, <cv::GArray<std::string>(
    cv::GArray<std::string>, TrackedIds, LostIds)>, "test.custom.trackClasses") {
    static cv::GArrayDesc outMeta(const cv::GArrayDesc&,
        const cv::GArrayDesc&,
        const cv::GArrayDesc&) {
        return cv::empty_array_desc();
    }
};

GAPI_OCV_KERNEL_ST(OCVTrackClasses, TrackClasses, TrackedClasses) {
    static void setup(cv::GArrayDesc, cv::GArrayDesc, cv::GArrayDesc,
        std::shared_ptr<TrackedClasses>&state,
        const cv::GCompileArgs&) {
        state = std::shared_ptr<TrackedClasses>(new TrackedClasses());
    }

    static void run(const std::vector<std::string>&tracked_classes,
        const std::vector<uint64_t>&tr_ids,
        const std::vector<uint64_t>&lost_ids,
        std::vector<std::string>&all_classes,
        TrackedClasses & state) {
        GAPI_Assert(tracked_classes.size() == tr_ids.size());

        all_classes.clear();

        // Remove lost ids
        for (const auto& id : lost_ids) {
            auto it = state.m_map.find(id);
            if (it == state.m_map.end()) {
                cv::util::throw_error(std::logic_error("Object is not being tracked"));
            }
            else {
                state.m_map.erase(it);
            }
        }

        // Add/change tracked classes
        for (size_t i = 0; i < tr_ids.size(); ++i) {
            state.m_map[tr_ids[i]] = tracked_classes[i];
        }

        // Get all objects from the state
        for (const auto& obj : state.m_map) {
            all_classes.push_back(obj.second);
        }
    }
};
// } //

TEST(VASObjectTracker, PipelineTest)
{
    constexpr int32_t frames_to_handle = 30;
    std::string pathToVideo = opencv_test::findDataFile("cv/video/768x576.avi", false);

    std::vector<std::vector<cv::Rect>> input_boxes(frames_to_handle);
    std::vector<std::vector<int>> input_boxes_ids(frames_to_handle);
    std::vector<std::vector<std::string>> input_labels(frames_to_handle);

    std::string path_to_boxes = opencv_test::findDataFile("cv/video/768x576.yml", false);
    cv::FileStorage fs_input_boxes(path_to_boxes, cv::FileStorage::READ);
    cv::FileNode fn_input_boxes = fs_input_boxes.root();
    for (auto it = fn_input_boxes.begin(); it != fn_input_boxes.end(); ++it) {
        cv::FileNode fn_frame = *it;
        std::string frame_name = fn_frame.name();
        int frame_no = std::stoi(frame_name.substr(frame_name.find("_") + 1));

        for (auto fit = fn_frame.begin(); fit != fn_frame.end(); ++fit) {
            cv::FileNode fn_box = *fit;

            cv::Rect box((double)fn_box["x"], (double)fn_box["y"],
                (double)fn_box["width"], (double)fn_box["height"]);
            input_boxes[frame_no].push_back(box);
            input_boxes_ids[frame_no].push_back(fn_box["id"]);
            input_labels[frame_no].push_back(fn_box["label"]);
        }
    }

    cv::GMat in;

    cv::GArray<cv::Rect> detections;
    cv::GArray<int> det_ids;
    std::tie(detections, det_ids) = CvVideo768x576_Detect::on(in);

    constexpr float delta_time = 0.055f;
    cv::GArray<cv::Rect> detections_to_update;
    cv::GArray<uint64_t> tracking_ids;
    cv::GArray<uint64_t> tracking_ids_to_remove;
    std::tie(detections_to_update, tracking_ids, tracking_ids_to_remove) =
        cv::gapi::ot::track(in, detections, det_ids, delta_time);

    // Filter out of bounds tracked objects
    cv::GArray<cv::Rect> filtered_tr_objs;
    cv::GArray<uint64_t> filtered_tr_ids;
    std::tie(filtered_tr_objs, filtered_tr_ids) = TrackedFilterOutOfBounds::on(detections_to_update,
        tracking_ids);

    // Run Inference for classifier on the passed ROIs of the frame
    cv::GArray<std::string> labels = CvVideo768x576_Classify::on(filtered_tr_objs);

    // Get all objects
    cv::GArray<std::string> complete_filtered_cls_labels = TrackClasses::on(labels,
        tracking_ids,
        tracking_ids_to_remove);

    cv::GComputation ccomp(cv::GIn(in), cv::GOut(complete_filtered_cls_labels));

    // Graph compilation for streaming mode:
    auto compiled =
        ccomp.compileStreaming(cv::compile_args(
            cv::gapi::combine(cv::gapi::kernels<OCV_CvVideo768x576_Detect,
                                                OCV_CvVideo768x576_Classify,
                                                OCVTrackClasses,
                                                OCVTrackedFilterOutOfBounds>(),
                              cv::gapi::ot::cpu::kernels()),
            opencv_test::FrameDetectionsParams{ 0, input_boxes, input_boxes_ids },
            opencv_test::FrameLabelsParams{ 0, input_boxes, input_labels }));

    EXPECT_TRUE(compiled);
    EXPECT_FALSE(compiled.running());

    compiled.setSource<cv::gapi::wip::GCaptureSource>(pathToVideo);

    // Start of streaming:
    compiled.start();
    EXPECT_TRUE(compiled.running());

    // Streaming:
    std::vector<std::string> out_class_labels;

    std::size_t counter { }, limit { 30 };
    while(compiled.pull(cv::gout(out_class_labels)) && (counter < limit)) {
         ++counter;
     }

     compiled.stop();

     EXPECT_FALSE(compiled.running());
}
} // namespace opencv_test
