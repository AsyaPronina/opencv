// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation


#include "../test_precomp.hpp"

#include <stdexcept>

#include <opencv2/gapi/imgproc.hpp>
#include <opencv2/gapi/gproto.hpp>
#include <opencv2/gapi/gcomputation.hpp>
#include <opencv2/gapi/gcompiled.hpp>
#include <opencv2/gapi/gkernel.hpp>

#include "compiler/gmodel.hpp"

#include "api/gcomputation_priv.hpp"
#include "compiler/gcompiled_priv.hpp"
#include "compiler/gcompiler.hpp"
#include "compiler/passes/passes.hpp"

#include "compiler/passes/pattern_matching.hpp"

#include "../common/gapi_tests_common.hpp"

#include "logger.hpp"

namespace opencv_test
{

// 3 normal cases, 2 corner cases

//for Data node: SHAPE
//for OP node: kernel name
//for edge: port

static void setDataStorageToNode(ade::NodeHandle& dataNode, cv::gimpl::GModel::Graph& graph, cv::gimpl::Data::Storage storage) {
    auto& data = graph.metadata(dataNode).get<cv::gimpl::Data>();
    data.storage = storage;
}

static ade::NodeHandle createInputDataNode(cv::gimpl::GModel::Graph& graph, const GShape shape) {
    auto inputDataNh =  cv::gimpl::GModel::mkDataNode(graph, shape);
    setDataStorageToNode(inputDataNh, graph, cv::gimpl::Data::Storage::INPUT);
    return inputDataNh;
}

static ade::NodeHandle createOutputDataNode(cv::gimpl::GModel::Graph& graph, const GShape shape) {
    auto outputDataNh =  cv::gimpl::GModel::mkDataNode(graph, shape);
    setDataStorageToNode(outputDataNh, graph, cv::gimpl::Data::Storage::OUTPUT);
    return outputDataNh;
}

static ade::NodeHandle createDataNode(cv::gimpl::GModel::Graph& graph, const GShape shape) {
    return cv::gimpl::GModel::mkDataNode(graph, shape);

}

static ade::NodeHandle createOpNode(cv::gimpl::GModel::Graph& graph, std::string kernelName,
                                    std::size_t inArgsCount, std::string island = "") {
    GKernel k{ kernelName, {}, {} };
    std::vector<GArg> args { };

    for (std::size_t i = 0; i < inArgsCount; ++i) {
        args.push_back(GArg{});
    }

    return cv::gimpl::GModel::mkOpNode(graph, k, args, island);
}


//Pattern
//          +-----------------------------------------------+
//          |                                               |
//+------+  |    XXXXXXXXXX      +------+      XXXXXXXXXX   |  +------+
//|      |  |   X          X     |      |     X          X  |  |      |
//| GMat +----->X filter2d X+--->+ GMat +---->X filter2d X+--->+ GMat |
//+------+  |    XXXXXXXXXX      +------+      XXXXXXXXXX   |  +------+
//          |                                               |
//          |    CPU                                        |
//          +-----------------------------------------------+

//Computation
//          +-----------------------------------------------------------------------------------------------+
//          |                                                                                               |
//+------+  |  XXXXXXXXXX     +------+    XXXXXXXXXX     +------+    XXXXXXXXXX     +------+    XXXXXXXXXX  |  +------+
//|      |  | X          X    |      |   X          X    |      |   X          X    |      |   X          X |  |      |
//| GMat +--->X  erode   X+-->+ GMat +-->X filter2d X+-->+ GMat +-->X filter2d X+-->+ GMat +-->X  dilate  X+-->+ GMat |
//+------+  |  XXXXXXXXXX     +------+    XXXXXXXXXX     +------+    XXXXXXXXXX     +------+    XXXXXXXXXX  |  +------+
//          |                                                                                               |
//          |                                 CPU                                                           |
//          +-----------------------------------------------------------------------------------------------+

//Add for in the beginning and in the end
TEST(PatternMatching, MatchChainInTheMiddle)
{
//----------------------------Pattern graph---------------------------
    std::unique_ptr<ade::Graph> adePatternGraphPtr(new ade::Graph);
    cv::gimpl::GModel::Graph patternGraph(*adePatternGraphPtr);
    cv::gimpl::GModel::init(patternGraph);

    auto firstPDataNh = createInputDataNode(patternGraph, GShape::GMAT);
    auto internPDataNh = createDataNode(patternGraph, GShape::GMAT);
    auto lastPDataNh = createOutputDataNode(patternGraph, GShape::GMAT);

    auto pFilter2dNh1 = createOpNode(patternGraph, "test.pm.filter2d", 1, "CPU");
    auto pFilter2dNh2 = createOpNode(patternGraph, "test.pm.filter2d", 1, "CPU");

    cv::gimpl::GModel::linkIn(patternGraph, pFilter2dNh1, firstPDataNh, 0);
    cv::gimpl::GModel::linkOut(patternGraph, pFilter2dNh1, internPDataNh, 0);
    cv::gimpl::GModel::linkIn(patternGraph, pFilter2dNh2, internPDataNh, 0);
    cv::gimpl::GModel::linkOut(patternGraph, pFilter2dNh2, lastPDataNh, 0);

    cv::gimpl::Protocol patternP;
    patternP.inputs = {};
    patternP.outputs = {};
    patternP.in_nhs = { firstPDataNh };
    patternP.out_nhs = { lastPDataNh };
    patternGraph.metadata().set(patternP);

    auto pPassCtx = ade::passes::PassContext{*adePatternGraphPtr};
    ade::passes::TopologicalSort()(pPassCtx);

//-------------------------------------------------------------------

//-------------------------GComputation graph------------------------
    std::unique_ptr<ade::Graph> adeCompGraphPtr(new ade::Graph);
    cv::gimpl::GModel::Graph compGraph(*adeCompGraphPtr);
    cv::gimpl::GModel::init(compGraph);

    auto firstCDataNh = createInputDataNode(compGraph, GShape::GMAT);
    auto internCDataNh1 = createDataNode(compGraph, GShape::GMAT);
    auto internCDataNh2 = createDataNode(compGraph, GShape::GMAT);
    auto internCDataNh3 = createDataNode(compGraph, GShape::GMAT);
    auto lastCDataNh = createOutputDataNode(compGraph, GShape::GMAT);

    auto cErodeNh = createOpNode(compGraph, "test.pm.erode", 1, "CPU");
    auto cFilter2dNh1 = createOpNode(compGraph, "test.pm.filter2d", 1, "CPU");
    auto cFilter2dNh2 = createOpNode(compGraph, "test.pm.filter2d", 1, "CPU");
    auto cDilateNh = createOpNode(compGraph, "test.pm.dilate", 1, "CPU");

    cv::gimpl::GModel::linkIn(compGraph, cErodeNh, firstCDataNh, 0);
    cv::gimpl::GModel::linkOut(compGraph, cErodeNh, internCDataNh1, 0);
    cv::gimpl::GModel::linkIn(compGraph, cFilter2dNh1, internCDataNh1, 0);
    cv::gimpl::GModel::linkOut(compGraph, cFilter2dNh1, internCDataNh2, 0);
    cv::gimpl::GModel::linkIn(compGraph, cFilter2dNh2, internCDataNh2, 0);
    cv::gimpl::GModel::linkOut(compGraph, cFilter2dNh2, internCDataNh3, 0);
    cv::gimpl::GModel::linkIn(compGraph, cDilateNh, internCDataNh3, 0);
    cv::gimpl::GModel::linkOut(compGraph, cDilateNh, lastCDataNh, 0);

    cv::gimpl::Protocol compP;
    compP.inputs = {};
    compP.outputs = {};
    compP.in_nhs = { firstCDataNh };
    compP.out_nhs = { lastCDataNh };
    compGraph.metadata().set(compP);

    auto cPassCtx = ade::passes::PassContext{*adeCompGraphPtr};
    ade::passes::TopologicalSort()(cPassCtx);
 //--------------------------------------------------------------------


//-----------------------Pattern Matching------------------------------
     cv::gapi::SubgraphMatch match = cv::gapi::findMatches(patternGraph, compGraph);
//---------------------------------------------------------------------

//--------------------Pattern Matching Verification--------------------
     auto inputDataNodesMatches = match.inputDataNodesMatches;
     auto firstOpNodesMatches = match.firstOpNodesMatches;
     auto lastOpNodesMatches = match.lastOpNodesMatches;
     auto outputDataNodesMatches = match.outputDataNodesMatches;

     auto compSubgraphInt = match.internalLayers;

     ASSERT_EQ(inputDataNodesMatches.size(), 1);
     auto inputDataNodeMatch = *inputDataNodesMatches.begin();
     ASSERT_EQ(inputDataNodeMatch.first, firstPDataNh);
     ASSERT_EQ(inputDataNodeMatch.second, internCDataNh1);

     ASSERT_EQ(firstOpNodesMatches.size(), 1);
     auto firstOpNodeMatch = *firstOpNodesMatches.begin();
     ASSERT_EQ(firstOpNodeMatch.first, pFilter2dNh1);
     ASSERT_EQ(firstOpNodeMatch.second, cFilter2dNh1);

     ASSERT_EQ(lastOpNodesMatches.size(), 1);
     auto lastOpNodeMatch = *lastOpNodesMatches.begin();
     ASSERT_EQ(lastOpNodeMatch.first, pFilter2dNh2);
     ASSERT_EQ(lastOpNodeMatch.second, cFilter2dNh2);

     ASSERT_EQ(compSubgraphInt.size(), 1);
     auto compInternNode = *compSubgraphInt.begin();
     ASSERT_EQ(internCDataNh2, compInternNode);
//---------------------------------------------------------------------
}
} // namespace opencv_test
