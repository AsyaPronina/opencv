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

        std::list<ade::NodeHandle> internalLayers;

        bool ok() const {
            return    !inputDataNodes.empty() && !startOpNodes.empty()
                   && !finishOpNodes.empty() && !outputDataNodes.empty();

        }

       S nodes() const {
           S allNodes {};

           for (auto it = inputDataNodes.begin(); it != inputDataNodes.end(); ++it) {
               allNodes.insert(it->second);
           }

           for (auto it = startOpNodes.begin(); it != startOpNodes.end(); ++it) {
               allNodes.insert(it->second);
           }

           for (auto it = finishOpNodes.begin(); it != finishOpNodes.end(); ++it) {
               allNodes.insert(it->second);
           }

           for (auto it = outputDataNodes.begin(); it != outputDataNodes.end(); ++it) {
               allNodes.insert(it->second);
           }

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
           std::vector<ade::NodeHandle> pIns;
           for (auto inDataNode : inputDataNodes) {
              pIns.push_back(inDataNode.second);
           }
           return pIns;
       }


       std::vector<ade::NodeHandle> protoOuts() {
           std::vector<ade::NodeHandle> pOuts;
           for (auto outDataNode : outputDataNodes) {
              pOuts.push_back(outDataNode.second);
           }
           return pOuts;
       }
    };

    GAPI_EXPORTS SubgraphMatch findMatches(cv::gimpl::GModel::Graph patternGraph, cv::gimpl::GModel::Graph compGraph);

} //namespace gimpl
} //namespace cv
#endif // OPENCV_GAPI_PATTERN_MATCHING_HPP
