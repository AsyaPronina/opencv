// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_PATTERN_MATCHING_HPP
#define OPENCV_GAPI_PATTERN_MATCHING_HPP

#include <functional> 
#include <unordered_set>
#include <unordered_map>
#include <list>

#include "opencv2/gapi/gcomputation.hpp"
#include "opencv2/gapi/gcompiled.hpp"
#include "opencv2/gapi/gkernel.hpp"

#include "api/gcomputation_priv.hpp"
#include "api/gcall_priv.hpp"
#include "api/gnode_priv.hpp"

#include "compiler/gcompiled_priv.hpp"
#include "compiler/gmodel.hpp"

#include <ade/graph.hpp>
#include <ade/typed_graph.hpp>

namespace cv {
namespace gapi {

    struct SubgraphMatch {
        struct NodeHandleHashFunction {
            size_t operator()(const ade::NodeHandle& nh) const
            {
                return std::hash<ade::Node*>()(nh.get());
            }
        };
        std::unordered_map<ade::NodeHandle, ade::NodeHandle, NodeHandleHashFunction> inputDataNodesMatches;
        std::unordered_map<ade::NodeHandle, ade::NodeHandle, NodeHandleHashFunction> firstOpNodesMatches;
        std::unordered_map<ade::NodeHandle, ade::NodeHandle, NodeHandleHashFunction> lastOpNodesMatches;
        std::unordered_map<ade::NodeHandle, ade::NodeHandle, NodeHandleHashFunction> outputDataNodesMatches;

        std::list<ade::NodeHandle> internalLayers;
    };

    GAPI_EXPORTS SubgraphMatch findMatches(cv::gimpl::GModel::Graph patternGraph, cv::gimpl::GModel::Graph compGraph);

} //namespace gapi
} //namespace cv
#endif // OPENCV_GAPI_PATTERN_MATCHING_HPP
