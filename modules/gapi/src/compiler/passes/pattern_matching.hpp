// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation


#ifndef OPENCV_GAPI_PATTERN_MATCHING_HPP
#define OPENCV_GAPI_PATTERN_MATCHING_HPP

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <list>

#include "compiler/gmodel.hpp"

namespace cv {
namespace gimpl {

    struct SubgraphMatch {
        using M =  std::unordered_map< ade::NodeHandle              // Pattern graph node
                                     , ade::NodeHandle              // Test graph node
                                     , ade::HandleHasher<ade::Node>
                                     >;
        using S =  std::unordered_set< ade::NodeHandle
                                     , ade::HandleHasher<ade::Node>
                                     >;
        M inputDataNodes;
        M startOpNodes;
        M finishOpNodes;
        M outputDataNodes;

        std::vector<ade::NodeHandle> inputTestDataNodes;
        std::vector<ade::NodeHandle> outputTestDataNodes;

        std::list<ade::NodeHandle> internalLayers;

        bool ok() const {
            return    !inputDataNodes.empty() && !startOpNodes.empty()
                   && !finishOpNodes.empty() && !outputDataNodes.empty()
                   && !inputTestDataNodes.empty() && !outputTestDataNodes.empty();

        }

       S nodes() const {
           S allNodes {};

           allNodes.insert(inputTestDataNodes.begin(), inputTestDataNodes.end());

           for (auto it = startOpNodes.begin(); it != startOpNodes.end(); ++it) {
               allNodes.insert(it->second);
           }

           for (auto it = finishOpNodes.begin(); it != finishOpNodes.end(); ++it) {
               allNodes.insert(it->second);
           }

           allNodes.insert(outputTestDataNodes.begin(), outputTestDataNodes.end());

           allNodes.insert(internalLayers.begin(), internalLayers.end());

           return allNodes;
       }

       S startOps() {
            S sOps;
            for (auto opNope : startOpNodes) {
               sOps.insert(opNope.second);
            }
            return sOps;
       }

       S finishOps() {
            S fOps;
            for (auto opNope : finishOpNodes) {
               fOps.insert(opNope.second);
            }
            return fOps;
       }

       std::vector<ade::NodeHandle> protoIns() {
           return inputTestDataNodes;
       }


       std::vector<ade::NodeHandle> protoOuts() {
           return outputTestDataNodes;
       }
    };

    GAPI_EXPORTS SubgraphMatch findMatches(const cv::gimpl::GModel::Graph& patternGraph,
                                           const cv::gimpl::GModel::Graph& compGraph);

} //namespace gimpl
} //namespace cv
#endif // OPENCV_GAPI_PATTERN_MATCHING_HPP
