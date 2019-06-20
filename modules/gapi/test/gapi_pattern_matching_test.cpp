// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "test_precomp.hpp"

#include <stdexcept>
#include <opencv2/gapi/pattern_matching.hpp>
#include "logger.hpp"

namespace opencv_test
{
    TEST(PatternMatching, TestInputDataNodesLessAndOnDifferentLevels)
    {
        cv::Mat mat(1920, 780, CV_8UC3), mat1(mat), result;
        cv::randu(mat, cv::Scalar::all(0), cv::Scalar::all(255));
        std::vector<unsigned char> data{ 1, 1, 1, 1, 1, 1, 1, 1, 1 };
        cv::Mat kernel(3, 3, CV_8UC1, data.data());

        //----------------------------pattern--------------------------------
        cv::GMat patternIn1, patternIn2;
        auto patternFiltered = cv::gapi::filter2D(patternIn1, 3, kernel);
        auto patternAdded = cv::gapi::add(patternIn2, patternFiltered, 1);

        cv::GComputation pattern(patternIn1, patternIn2, patternAdded);
        // const auto& patternGraph = pattern.compile(cv::descr_of(mat)).priv().model();
        pattern.apply(cv::gin(mat, mat1), cv::gout(result));
        //-------------------------------------------------------------------

        //-------------------------input GComputation------------------------
        cv::Mat result1, result2, result3;
        cv::GMat in;
        auto filtered = cv::gapi::filter2D(in, 3, kernel);
        auto filteredFiltered = cv::gapi::filter2D(filtered, 3, kernel);
        auto added = cv::gapi::add(filtered, filteredFiltered, 1);

        cv::GComputation computation(cv::GIn(in), cv::GOut(added));
        // const auto& computationGraph = computation.compile(cv::descr_of(cv::gin(mat))).priv().model();
        computation.apply(cv::gin(mat), cv::gout(result1));
        //--------------------------------------------------------------------

        //-----------------------pattern matching-----------------------------
        // cv::gapi::findMatches(patternGraph, computationGraph);
        auto res = cv::gapi::findMatches(pattern.priv().m_lastCompiled.priv().model(), computation.priv().m_lastCompiled.priv().model());
    }


    TEST(PatternMatching, CSCResizeSplit3Substitute)
    {
        cv::Mat mat(1920, 780, CV_8UC3), mat1(mat), result;
        cv::randu(mat, cv::Scalar::all(0), cv::Scalar::all(255));
        std::vector<unsigned char> data{ 1, 1, 1, 1, 1, 1, 1, 1, 1 };
        cv::Mat kernel(3, 3, CV_8UC1, data.data());

        //----------------------------pattern--------------------------------
        cv::GMat patternIn1, patternIn2;
        auto patternFiltered = cv::gapi::filter2D(patternIn1, 3, kernel);
        auto patternAdded = cv::gapi::add(patternIn2, patternFiltered, 1);

        cv::GComputation pattern(patternIn1, patternIn2, patternAdded);
        // const auto& patternGraph = pattern.compile(cv::descr_of(mat)).priv().model();
        pattern.apply(cv::gin(mat, mat1), cv::gout(result));
        //-------------------------------------------------------------------

        //-------------------------input GComputation------------------------
        cv::Mat result1, result2, result3;
        cv::GMat in;
        auto filtered = cv::gapi::filter2D(in, 3, kernel);
        auto filteredFiltered = cv::gapi::filter2D(filtered, 3, kernel);
        auto added = cv::gapi::add(filtered, filteredFiltered, 1);

        cv::GComputation computation(cv::GIn(in), cv::GOut(added));
        // const auto& computationGraph = computation.compile(cv::descr_of(cv::gin(mat))).priv().model();
        computation.apply(cv::gin(mat), cv::gout(result1));
        //--------------------------------------------------------------------

        //-----------------------pattern matching-----------------------------
        // cv::gapi::findMatches(patternGraph, computationGraph);
        auto res = cv::gapi::findMatches(pattern.priv().m_lastCompiled.priv().model(), computation.priv().m_lastCompiled.priv().model());
    }
//--------------------------------------------------------------------
} // namespace opencv_test
