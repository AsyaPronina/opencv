// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation


#ifndef OPENCV_GAPI_PATTERN_MATCHING_HPP
#define OPENCV_GAPI_PATTERN_MATCHING_HPP

#include <unordered_map>
#include <list>

#include "compiler/gmodel.hpp"

namespace cv {
namespace gapi {

    struct SubgraphMatch {
        std::unordered_map<ade::NodeHandle, ade::NodeHandle, ade::HandleHasher<ade::Node>> inputDataNodesMatches;
        std::unordered_map<ade::NodeHandle, ade::NodeHandle, ade::HandleHasher<ade::Node>> firstOpNodesMatches;
        std::unordered_map<ade::NodeHandle, ade::NodeHandle, ade::HandleHasher<ade::Node>> lastOpNodesMatches;
        std::unordered_map<ade::NodeHandle, ade::NodeHandle, ade::HandleHasher<ade::Node>> outputDataNodesMatches;

        std::list<ade::NodeHandle> internalLayers;
    };

    GAPI_EXPORTS SubgraphMatch findMatches(cv::gimpl::GModel::Graph patternGraph, cv::gimpl::GModel::Graph compGraph);

} //namespace gapi
} //namespace cv
#endif // OPENCV_GAPI_PATTERN_MATCHING_HPP
