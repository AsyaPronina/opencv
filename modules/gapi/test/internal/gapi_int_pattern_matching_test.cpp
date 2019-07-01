// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation


#include "../test_precomp.hpp"

#include <stdexcept>

#include "compiler/gmodel.hpp"
#include "compiler/gmodel_priv.hpp"

#include "api/gcomputation_priv.hpp"
#include "compiler/gcompiler.hpp"
#include "compiler/gmodelbuilder.hpp"
#include "compiler/passes/passes.hpp"

#include "compiler/passes/pattern_matching.hpp"

#include "../common/gapi_tests_common.hpp"

#include "logger.hpp"

namespace opencv_test
{

namespace matching_test {
namespace  {
using V = std::vector<ade::NodeHandle>;
using S =  std::unordered_set< ade::NodeHandle
                             , ade::HandleHasher<ade::Node>
                             >;

void initGModel(ade::Graph& gr,
                cv::GProtoInputArgs&& in,
                cv::GProtoOutputArgs&& out) {

    cv::gimpl::GModel::Graph gm(gr);
    cv::gimpl::GModel::init(gm);
    auto proto_slots = cv::gimpl::GModelBuilder(gr)
            .put(in.m_args, out.m_args);

    cv::gimpl::Protocol p;
    std::tie(p.inputs, p.outputs, p.in_nhs, p.out_nhs) = proto_slots;
    gm.metadata().set(p);
}

bool isConsumedBy(cv::gimpl::GModel::Graph gm, ade::NodeHandle data_nh, ade::NodeHandle op_nh) {
    auto oi = cv::gimpl::GModel::orderedInputs(gm, op_nh);
    return std::find(oi.begin(), oi.end(), data_nh) != oi.end();
}

std::string opName(cv::gimpl::GModel::Graph gm, ade::NodeHandle op_nh) {
    return gm.metadata(op_nh).get<cv::gimpl::Op>().k.name;
}

}
} // matching_test

TEST(PatternMatching, TestSimple1)
{
    // Pattern
    ade::Graph pg;
    {
        GMat in;
        GMat out = cv::gapi::bitwise_not(in);
        matching_test::initGModel(pg, cv::GIn(in), cv::GOut(out));
   }

    // Test
    ade::Graph tg;
    GMat in;
    GMat out = cv::gapi::bitwise_not(in);
    matching_test::initGModel(tg, cv::GIn(in), cv::GOut(out));

    // Pattern Matching
    cv::gimpl::GModel::Graph pgm(pg);
    cv::gimpl::GModel::Graph tgm(tg);
    cv::gimpl::SubgraphMatch match = cv::gimpl::findMatches(pg, tg);

    // Inspecting results:
    EXPECT_TRUE(match.ok());

    auto nodes = match.nodes();
    EXPECT_EQ(3u, nodes.size());

    const auto in_nh = cv::gimpl::GModel::dataNodeOf(tgm, in);
    const auto out_nh = cv::gimpl::GModel::dataNodeOf(tgm, out);
    const auto op_nh = cv::gimpl::GModel::producerOf(tgm, out_nh);

    EXPECT_EQ(matching_test::S({in_nh, out_nh, op_nh}), nodes);
    EXPECT_EQ(cv::gapi::core::GNot::id(), matching_test::opName(tgm, op_nh));
    EXPECT_TRUE(matching_test::isConsumedBy(tgm, in_nh, op_nh));
    EXPECT_EQ(matching_test::S{op_nh}, match.startOps());
    EXPECT_EQ(matching_test::S{op_nh}, match.finishOps());
    EXPECT_EQ(matching_test::V{in_nh}, match.protoIns());
    EXPECT_EQ(matching_test::V{out_nh}, match.protoOuts());
}

TEST(PatternMatching, TestSimple2)
{
    // Pattern
    ade::Graph pg;
    {
        GMat in;
        GMat out = cv::gapi::bitwise_not(in);
        matching_test::initGModel(pg, cv::GIn(in), cv::GOut(out));
   }

    // Test
    ade::Graph tg;
    GMat in;
    GMat tmp = cv::gapi::bitwise_not(in);
    GMat out = cv::gapi::blur(tmp, cv::Size(3, 3));
    matching_test::initGModel(tg, cv::GIn(in), cv::GOut(out));

    // Pattern Matching
    cv::gimpl::GModel::Graph pgm(pg);
    cv::gimpl::GModel::Graph tgm(tg);
    cv::gimpl::SubgraphMatch match = cv::gimpl::findMatches(pg, tg);

    // Inspecting results:
    EXPECT_TRUE(match.ok());

    auto nodes = match.nodes();
    EXPECT_EQ(3u, nodes.size());

    const auto in_nh = cv::gimpl::GModel::dataNodeOf(tgm, in);
    const auto tmp_nh = cv::gimpl::GModel::dataNodeOf(tgm, tmp);
    const auto op_nh = cv::gimpl::GModel::producerOf(tgm, tmp_nh);

    EXPECT_EQ(matching_test::S({in_nh, tmp_nh, op_nh}), nodes);
    EXPECT_EQ(cv::gapi::core::GNot::id(), matching_test::opName(tgm, op_nh));
    EXPECT_TRUE(matching_test::isConsumedBy(tgm, in_nh, op_nh));
    EXPECT_EQ(matching_test::S{op_nh}, match.startOps());
    EXPECT_EQ(matching_test::S{op_nh}, match.finishOps());
    EXPECT_EQ(matching_test::V{in_nh}, match.protoIns());
    EXPECT_EQ(matching_test::V{tmp_nh}, match.protoOuts());
}

TEST(PatternMatching, TestSimple3)
{
    // Pattern
    ade::Graph pg;
    {
        GMat in;
        GMat out = cv::gapi::bitwise_not(in);
        matching_test::initGModel(pg, cv::GIn(in), cv::GOut(out));
   }

    // Test
    ade::Graph tg;
    GMat in;
    GMat tmp = cv::gapi::blur(in, cv::Size(3, 3));
    GMat out = cv::gapi::bitwise_not(tmp);
    matching_test::initGModel(tg, cv::GIn(in), cv::GOut(out));

    // Pattern Matching
    cv::gimpl::GModel::Graph pgm(pg);
    cv::gimpl::GModel::Graph tgm(tg);
    cv::gimpl::SubgraphMatch match = cv::gimpl::findMatches(pg, tg);

    // Inspecting results:
    EXPECT_TRUE(match.ok());

    auto nodes = match.nodes();
    EXPECT_EQ(3u, nodes.size());

    const auto tmp_nh = cv::gimpl::GModel::dataNodeOf(tgm, tmp);
    const auto out_nh = cv::gimpl::GModel::dataNodeOf(tgm, out);
    const auto op_nh = cv::gimpl::GModel::producerOf(tgm, out_nh);

    EXPECT_EQ(matching_test::S({tmp_nh, out_nh, op_nh}), nodes);
    EXPECT_EQ(cv::gapi::core::GNot::id(), matching_test::opName(tgm, op_nh));
    EXPECT_TRUE(matching_test::isConsumedBy(tgm, tmp_nh, op_nh));
    EXPECT_EQ(matching_test::S{op_nh}, match.startOps());
    EXPECT_EQ(matching_test::S{op_nh}, match.finishOps());
    EXPECT_EQ(matching_test::V{tmp_nh}, match.protoIns());
    EXPECT_EQ(matching_test::V{out_nh}, match.protoOuts());
}

TEST(PatternMatching, TestMultiplePatternOuts)
{
    // Pattern
    ade::Graph pg;
    {
        GMat in;
        GMat dx, dy;
        std::tie(dx, dy) = cv::gapi::SobelXY(in, -1, 1);
        matching_test::initGModel(pg, cv::GIn(in), cv::GOut(dx, dy));
   }

    // Test
    ade::Graph tg;
    GMat in;
    GMat dx, dy;
    std::tie(dx, dy) = cv::gapi::SobelXY(in, -1, 1);
    matching_test::initGModel(tg, cv::GIn(in), cv::GOut(dx, dy));

    // Pattern Matching
    cv::gimpl::GModel::Graph pgm(pg);
    cv::gimpl::GModel::Graph tgm(tg);
    cv::gimpl::SubgraphMatch match = cv::gimpl::findMatches(pg, tg);

    // Inspecting results:
    EXPECT_TRUE(match.ok());

    auto nodes = match.nodes();
    EXPECT_EQ(4, nodes.size());

    const auto in_nh = cv::gimpl::GModel::dataNodeOf(tgm, in);
    const auto dx_nh = cv::gimpl::GModel::dataNodeOf(tgm, dx);
    const auto dy_nh = cv::gimpl::GModel::dataNodeOf(tgm, dy);
    const auto op_nh = cv::gimpl::GModel::producerOf(tgm, dx_nh);
    EXPECT_EQ(op_nh,  cv::gimpl::GModel::producerOf(tgm, dy_nh));

    EXPECT_EQ(matching_test::S({in_nh, dx_nh, dy_nh, op_nh}), nodes);
    EXPECT_EQ(cv::gapi::imgproc::GSobelXY::id(), matching_test::opName(tgm, op_nh));
    EXPECT_TRUE(matching_test::isConsumedBy(tgm, in_nh, op_nh));
    EXPECT_EQ(matching_test::S{op_nh}, match.startOps());
    EXPECT_EQ(matching_test::S{op_nh}, match.finishOps());
    EXPECT_EQ(matching_test::V{in_nh}, match.protoIns());
    EXPECT_EQ(matching_test::V({dx_nh, dy_nh}), match.protoOuts());
}

//FIXME: To switch from filter2d kernel (which shall be matched by params too) to another one
TEST(PatternMatching, MatchChainInTheMiddle)
{
    // Pattern
    ade::Graph pg;
    {
        GMat in;
        GMat tmp = cv::gapi::filter2D(in, -1, {});
        GMat out = cv::gapi::filter2D(tmp, -1, {});
        matching_test::initGModel(pg, cv::GIn(in), cv::GOut(out));
   }

    // Test
    ade::Graph tg;
    GMat in;
    GMat tmp1 = cv::gapi::erode3x3(in);
    GMat tmp2 = cv::gapi::filter2D(tmp1, -1, {});
    GMat tmp3 = cv::gapi::filter2D(tmp2, -1, {});
    GMat out = cv::gapi::dilate3x3(tmp3);
    matching_test::initGModel(tg, cv::GIn(in), cv::GOut(out));

    // Pattern Matching
    cv::gimpl::GModel::Graph pgm(pg);
    cv::gimpl::GModel::Graph tgm(tg);
    cv::gimpl::SubgraphMatch match = cv::gimpl::findMatches(pg, tg);

    // Inspecting results:
    EXPECT_TRUE(match.ok());

    auto nodes = match.nodes();
    EXPECT_EQ(5u, nodes.size());

    const auto tmp1_nh = cv::gimpl::GModel::dataNodeOf(tgm, tmp1);
    const auto tmp2_nh = cv::gimpl::GModel::dataNodeOf(tgm, tmp2);
    const auto tmp3_nh = cv::gimpl::GModel::dataNodeOf(tgm, tmp3);
    const auto op1_nh = cv::gimpl::GModel::producerOf(tgm, tmp2_nh); // 1st filter2D
    const auto op2_nh = cv::gimpl::GModel::producerOf(tgm, tmp3_nh); // 2nd filter2D

    EXPECT_EQ(matching_test::S({tmp1_nh, tmp2_nh, tmp3_nh, op1_nh, op2_nh}), nodes);

    EXPECT_EQ(cv::gapi::imgproc::GFilter2D::id(), matching_test::opName(tgm, op1_nh));
    EXPECT_EQ(cv::gapi::imgproc::GFilter2D::id(), matching_test::opName(tgm, op2_nh));

    EXPECT_EQ(1u, tmp2_nh->outEdges().size());
    EXPECT_TRUE(matching_test::isConsumedBy(tgm, tmp1_nh, op1_nh));
    EXPECT_TRUE(matching_test::isConsumedBy(tgm, tmp2_nh, op2_nh));

    EXPECT_EQ(matching_test::S({op1_nh}), match.startOps());
    EXPECT_EQ(matching_test::S({op2_nh}), match.finishOps());
    EXPECT_EQ(matching_test::V{ tmp1_nh }, match.protoIns());
    EXPECT_EQ(matching_test::V{ tmp3_nh }, match.protoOuts());
}

TEST(PatternMatching, TestPreproc)
{
    // Pattern
    ade::Graph pg;
    {
        GMat in;
        GMat tmp = cv::gapi::resize(in, cv::Size{224, 224});
        GMat b, g, r;
        std::tie(b, g, r) = cv::gapi::split3(tmp);
        matching_test::initGModel(pg, cv::GIn(in), cv::GOut(b, g, r));
    }

    // Test
    ade::Graph tg;
    GMat y, uv;
    GMat bgr = cv::gapi::NV12toBGR(y, uv);
    GMat tmp = cv::gapi::resize(bgr, cv::Size{224, 224});
    GMat b, g, r;
    std::tie(b, g, r) = cv::gapi::split3(tmp);
    matching_test::initGModel(tg, cv::GIn(y, uv), cv::GOut(b, g, r));

    // Pattern Matching
    cv::gimpl::GModel::Graph pgm(pg);
    cv::gimpl::GModel::Graph tgm(tg);
    cv::gimpl::SubgraphMatch match = cv::gimpl::findMatches(pg, tg);

    // Inspecting results:
    EXPECT_TRUE(match.ok());

//    auto nodes = match.nodes();
//    EXPECT_EQ(5u, nodes.size());

//    const auto tmp1_nh = cv::gimpl::GModel::dataNodeOf(tgm, tmp1);
//    const auto tmp2_nh = cv::gimpl::GModel::dataNodeOf(tgm, tmp2);
//    const auto tmp3_nh = cv::gimpl::GModel::dataNodeOf(tgm, tmp3);
//    const auto op1_nh = cv::gimpl::GModel::producerOf(tgm, tmp2_nh); // 1st filter2D
//    const auto op2_nh = cv::gimpl::GModel::producerOf(tgm, tmp3_nh); // 2nd filter2D

//    EXPECT_EQ(matching_test::S({tmp1_nh, tmp2_nh, tmp3_nh, op1_nh, op2_nh}), nodes);

//    EXPECT_EQ(cv::gapi::imgproc::GFilter2D::id(), matching_test::opName(tgm, op1_nh));
//    EXPECT_EQ(cv::gapi::imgproc::GFilter2D::id(), matching_test::opName(tgm, op2_nh));

//    EXPECT_EQ(1u, tmp2_nh->outEdges().size());
//    EXPECT_TRUE(matching_test::isConsumedBy(tgm, tmp1_nh, op1_nh));
//    EXPECT_TRUE(matching_test::isConsumedBy(tgm, tmp2_nh, op2_nh));

//    EXPECT_EQ(matching_test::S({op1_nh}), match.startOps());
//    EXPECT_EQ(matching_test::S({op2_nh}), match.finishOps());
//    EXPECT_EQ(matching_test::V{ tmp1_nh }, match.protoIns());
//    EXPECT_EQ(matching_test::V{ tmp3_nh }, match.protoOuts());
}

TEST(PatternMatching, CheckNoMatch)
{
    // Pattern
    ade::Graph pg;
    {
        GMat in;
        GMat tmp = cv::gapi::filter2D(in, -1, {});
        GMat out = cv::gapi::filter2D(tmp, -1, {});
        matching_test::initGModel(pg, cv::GIn(in), cv::GOut(out));
   }

    // Test
    ade::Graph tg;
    {
        GMat in;
        GMat tmp1 = cv::gapi::erode3x3(in);
        GMat out = cv::gapi::dilate3x3(tmp1);
        matching_test::initGModel(tg, cv::GIn(in), cv::GOut(out));
    }

    // Pattern Matching
    cv::gimpl::GModel::Graph pgm(pg);
    cv::gimpl::GModel::Graph tgm(tg);
    cv::gimpl::SubgraphMatch match = cv::gimpl::findMatches(pg, tg);

    // Inspecting results:
    EXPECT_FALSE(match.ok());
}

} // namespace opencv_test
