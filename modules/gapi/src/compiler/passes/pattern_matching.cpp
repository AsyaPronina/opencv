// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation

#include <unordered_set>

#include "pattern_matching.hpp"

cv::gimpl::SubgraphMatch cv::gimpl::findMatches(cv::gimpl::GModel::Graph patternGraph, cv::gimpl::GModel::Graph compGraph) {
    using Graph = cv::gimpl::GModel::Graph;
    using Metadata = typename Graph::MetadataT;
    using VisitedMatchings = std::list<std::pair<ade::NodeHandle, ade::NodeHandle>>;

    //TODO: Possibly, we may add N^2 check whether this graph may match or not at all.
    //      Check that all pattern OP nodes exist in computational graph.

    std::unordered_set<ade::NodeHandle, ade::HandleHasher<ade::Node>> firstPatternOpNodes;
    std::unordered_set<ade::NodeHandle, ade::HandleHasher<ade::Node>> lastPatternOpNodes;

    auto firstPatternDataNodes = patternGraph.metadata().get<cv::gimpl::Protocol>().in_nhs;
    auto lastPatternDataNodes = patternGraph.metadata().get<cv::gimpl::Protocol>().out_nhs;

    for (auto node : firstPatternDataNodes) {
        auto opNodes = node->outNodes();
        firstPatternOpNodes.insert(opNodes.begin(), opNodes.end());
    }

    for (auto node : lastPatternDataNodes) {
        auto opNodes = node->inNodes();
        lastPatternOpNodes.insert(opNodes.begin(), opNodes.end());
    }

    // For visited matchings to work correctly it is required that output nodes from some node found in both graphs were in the same order!
    // And that outNodes array contains these nodes right in the order that they are presented in graph (left-to-right nodes-exact).
    // Else graphs were not assumed to be equal.
    VisitedMatchings matchedVisitedNodes;

    auto dataNodesComparator = [](std::pair<const ade::NodeHandle, std::vector<std::size_t>> first, Metadata firstMetadata, std::pair<const ade::NodeHandle, std::vector<std::size_t>> second, Metadata secondMetadata) {
        if (secondMetadata.get<cv::gimpl::NodeType>().t != cv::gimpl::NodeType::DATA) {
            //TODO: FIX ASAP
            throw std::logic_error("NodeType of passed node as second argument shall be NodeType::DATA!");
        }

        if (firstMetadata.get<cv::gimpl::Data>().shape != secondMetadata.get<cv::gimpl::Data>().shape) {
            return false;
        }

        //todo
        std::sort(first.second.begin(), first.second.end());
        std::sort(second.second.begin(), second.second.end());
        if (first.second != second.second) {
            return false;
        }

        auto firstOutputEdges = first.first->outEdges();
        auto secondOutputEdges = second.first->outEdges();

        if (firstOutputEdges.size() != secondOutputEdges.size()) {
            return false;
        }

        return true;
    };

    auto opNodesComparator = [&matchedVisitedNodes](std::pair<const ade::NodeHandle, std::vector<std::size_t>> first, Metadata firstMetadata, std::pair<const ade::NodeHandle, std::vector<std::size_t>> second, Metadata secondMetadata, bool& isAlreadyVisited) {
        if (secondMetadata.get<cv::gimpl::NodeType>().t != cv::gimpl::NodeType::OP) {
            //throw std::logic_error("NodeType of passed node as second argument shall be NodeType::OP!");
            //TODO: FIX ASAP
            return false;
        }

        // Assuming that if kernels names are the same then output DATA nodes counts from kernels are the same.
        // Assuming that if kernels names are the same then input DATA nodes counts to kernels are the same.
        if (firstMetadata.get<cv::gimpl::Op>().k.name != secondMetadata.get<cv::gimpl::Op>().k.name) {
            return false;
        }

        // Extra for our case, because we can't create graph contained operation, which has multiple returns and all them are located in 1 variable (in 1 DATA node).
        auto firstOutputNodes = first.first->outNodes();
        auto secondOutputNodes = second.first->outNodes();

        if (firstOutputNodes.size() != secondOutputNodes.size()) {
            return false;
        }
        // extra

        //todo
        std::sort(first.second.begin(), first.second.end());
        std::sort(second.second.begin(), second.second.end());
        if (first.second != second.second) {
            return false;
        }
        ;
        //shall be done another way.
        auto foundit = std::find_if(matchedVisitedNodes.begin(), matchedVisitedNodes.end(), [&first](std::pair<ade::NodeHandle, ade::NodeHandle> match) { return first.first == match.first; });
        if (foundit != matchedVisitedNodes.end()) {
            if (second.first != foundit->second) {
                return false;
            }

            isAlreadyVisited = true;
        }

        return true;
    };

    std::unordered_map<ade::NodeHandle, std::vector<ade::NodeHandle>, ade::HandleHasher<ade::Node>> allMatchingsForFirstOpNodes;
    std::size_t possibleStartPointsCount = 1;

    auto compNodes = compGraph.nodes();
    for (auto firstPatternOpNode : firstPatternOpNodes) {
        std::vector<ade::NodeHandle> possibleMatchings;
        std::copy_if(compNodes.begin(), compNodes.end(), std::back_inserter(possibleMatchings), [&firstPatternOpNode, &patternGraph, &compGraph, &opNodesComparator](const ade::NodeHandle& node) {
            auto firstMetadata = patternGraph.metadata(firstPatternOpNode);
            auto secondMetadata = compGraph.metadata(node);
            bool stub = false;
            /* TODO: FIXX */
            return opNodesComparator(std::make_pair(firstPatternOpNode, std::vector<std::size_t>{ 0ul }), firstMetadata, std::make_pair(node, std::vector<std::size_t>{ 0ul }), secondMetadata, stub);
        });

        allMatchingsForFirstOpNodes[firstPatternOpNode] = possibleMatchings;
        possibleStartPointsCount *= possibleMatchings.size();
    }

    // Bad namings
    // TODO FIX: Use using
    std::unordered_map<ade::NodeHandle, ade::NodeHandle, ade::HandleHasher<ade::Node>> subgraphIns;
    std::unordered_map<ade::NodeHandle, ade::NodeHandle, ade::HandleHasher<ade::Node>> subgraphOuts;
    std::list<ade::NodeHandle> subgraphInternals;


    // Structural matching first, semantic matching second.

    //TODO: found, think on naming
    bool notFound = true;
    std::size_t i = 0;
    while (notFound && (i < possibleStartPointsCount)) {
        subgraphIns.clear();
        subgraphOuts.clear();
        subgraphInternals.clear();
        matchedVisitedNodes.clear();

        std::size_t div = i;
        for (auto allMatchingsForFirstOpNode : allMatchingsForFirstOpNodes) { //order is not determined: for ex., for last node. =( use ordered set and map to ensure order.
            auto size = allMatchingsForFirstOpNode.second.size();
            std::size_t index = div % size;
            div = div / size;
            auto firstCompOpNode = allMatchingsForFirstOpNode.second[index];
            matchedVisitedNodes.push_back({ allMatchingsForFirstOpNode.first, firstCompOpNode });
            //subgraphIns.push_back(matchedVisitedNodes.back());
        }

        //think on naming (stop instead)
        bool nonStop = true;
        bool isSearchFailed = false;

        typename VisitedMatchings::iterator matchIt = matchedVisitedNodes.begin();
        std::size_t size = matchedVisitedNodes.size();
        std::size_t index = 0;

        while (nonStop) {
            for (; index < size && !isSearchFailed; ++index, ++matchIt) {

                bool cond1 = std::find(lastPatternOpNodes.begin(), lastPatternOpNodes.end(), matchIt->first) != lastPatternOpNodes.end();
                if (cond1) {
                    subgraphOuts[matchIt->first] = matchIt->second;
                }

                bool cond2 = std::find(firstPatternOpNodes.begin(), firstPatternOpNodes.end(), matchIt->first) != firstPatternOpNodes.end();
                if (cond2) {
                    subgraphIns[matchIt->first] = matchIt->second;
                }

                if (!cond1 && !cond2) {
                    subgraphInternals.push_back(matchIt->second);
                }

                std::unordered_map<ade::NodeHandle, std::vector<std::size_t>, ade::HandleHasher<ade::Node>> patternOutputNodesLabeled;
                std::unordered_map<ade::NodeHandle, std::vector<std::size_t>, ade::HandleHasher<ade::Node>> compOutputNodesLabeled;

                auto patternOutputEdges = matchIt->first->outEdges();
                auto compOutputEdges = matchIt->second->outEdges();

                auto addLabelToNode = [](ade::NodeHandle node, ade::EdgeHandle edge, const Graph& graph, std::unordered_map<ade::NodeHandle, std::vector<size_t>, ade::HandleHasher<ade::Node>>& labeledNodes) {
                    if (graph.metadata(node).get<cv::gimpl::NodeType>().t == cv::gimpl::NodeType::OP) {
                        labeledNodes[node].push_back(graph.metadata(edge).get<cv::gimpl::Input>().port);
                    }
                    else {
                        //Ruslan: use 1 int instead of vector as for DATA node we can have only 1 input edge.
                        labeledNodes[node].push_back(graph.metadata(edge).get<cv::gimpl::Output>().port);
                    }
                };

                for (auto patternOutputEdge : patternOutputEdges) {
                    if (!patternOutputEdge->dstNode()->outEdges().empty()) {
                        //Assuming that there is no case for the op node without output data nodes.
                        addLabelToNode(patternOutputEdge->dstNode(), patternOutputEdge, patternGraph, patternOutputNodesLabeled);
                    }
                }

                for (auto compOutputEdge : compOutputEdges) {
                         addLabelToNode(compOutputEdge->dstNode(), compOutputEdge, compGraph, compOutputNodesLabeled);
                }

                for (auto patternIt = patternOutputNodesLabeled.begin(); patternIt != patternOutputNodesLabeled.end(); ++patternIt) {
                    bool isAlreadyVisited = false;

                    auto matchedIt = std::find_if(compOutputNodesLabeled.begin(), compOutputNodesLabeled.end(),
                        [&patternIt, &patternGraph, &compGraph, &dataNodesComparator, &opNodesComparator, &isAlreadyVisited](std::pair<const ade::NodeHandle, std::vector<std::size_t>>& compNode) -> bool {
                        auto patternNodeMetadata = patternGraph.metadata(patternIt->first);
                        auto compNodeMetadata = compGraph.metadata(compNode.first);

                        if (patternNodeMetadata.get<cv::gimpl::NodeType>().t == cv::gimpl::NodeType::DATA) {
                            return dataNodesComparator(*patternIt, patternNodeMetadata, compNode, compNodeMetadata);
                        }
                        else {
                            return opNodesComparator(*patternIt, patternNodeMetadata, compNode, compNodeMetadata, isAlreadyVisited);
                        }
                    });

                    if (matchedIt == compOutputNodesLabeled.end()) {
                        nonStop = false;
                        isSearchFailed = true;
                        break;
                    }

                    //We shall not put in the matchings already visited nodes.
                    if (!isAlreadyVisited) {
                        matchedVisitedNodes.push_back({ patternIt->first, matchedIt->first });
                    }
                }
            }

            // 2. Secondly, update the matching array
            // Ruslan: no is
            if (!isSearchFailed) {
                if (std::distance(matchIt, matchedVisitedNodes.end()) == 0) {
                    //Found
                    nonStop = false;
                    notFound = false;
                }

                index = 0;
                size = std::distance(matchIt, matchedVisitedNodes.end());
            }
        }

        ++i;
    }

    std::unordered_map<ade::NodeHandle, ade::NodeHandle, ade::HandleHasher<ade::Node>> inputApiMatch(firstPatternDataNodes.size());
    std::unordered_map<ade::NodeHandle, ade::NodeHandle, ade::HandleHasher<ade::Node>> outputApiMatch(lastPatternDataNodes.size());

    bool matched = true;

    if (!notFound) {
        VisitedMatchings matchedVisitedFirstDataNodes;
        for (auto it = subgraphIns.begin(); it != subgraphIns.end() && matched; ++it) {
            auto match = *it;
            auto patternInputEdges = match.first->inEdges();
            auto compInputEdges = match.second->inEdges();

            if (match.first->inNodes().size() < match.second->inNodes().size()) {
                matchedVisitedFirstDataNodes.clear();
                inputApiMatch.clear();
                matched = false;
                break;
            }

            for (auto patternIt = patternInputEdges.begin(); patternIt != patternInputEdges.end(); ++patternIt) {

                if ((*patternIt)->srcNode()->inEdges().size() != 0) {
                    continue;
                }

                auto matchedIt = std::find_if(compInputEdges.begin(), compInputEdges.end(),
                    [&patternIt, &patternGraph, &compGraph, &matchedVisitedFirstDataNodes](const ade::EdgeHandle& compEdge) -> bool {
                    auto patternInputPort = patternGraph.metadata(*patternIt).get<cv::gimpl::Input>().port;
                    auto compInputPort = compGraph.metadata(compEdge).get<cv::gimpl::Input>().port;

                    if (patternInputPort != compInputPort) {
                        return false;
                    }

                    auto foundit = std::find_if(matchedVisitedFirstDataNodes.begin(), matchedVisitedFirstDataNodes.end(), [&patternIt](std::pair<ade::NodeHandle, ade::NodeHandle> matchedNodes) { return (*patternIt)->srcNode() == matchedNodes.first; });
                    if (foundit != matchedVisitedFirstDataNodes.end()) {
                        if (compEdge->srcNode() != foundit->second) {
                            return false;
                        }
                    }

                    //shall be map in this case as doesn't require iterations during modification
                    //Get rid of it and use inputApiMatch
                    matchedVisitedFirstDataNodes.push_back({ (*patternIt)->srcNode(), compEdge->srcNode() });

                    return true;
                });

                if (matchedIt == compInputEdges.end()) {
                    matchedVisitedFirstDataNodes.clear();
                    inputApiMatch.clear();
                    matched = false;
                    break;
                }
                inputApiMatch[(*patternIt)->srcNode()] = (*matchedIt)->srcNode();
            }

        }

        if (matched) {
            std::unordered_set<ade::NodeHandle, ade::HandleHasher<ade::Node>> visitedLastDataNodes;
            for (auto it = subgraphOuts.begin(); it != subgraphOuts.end() && matched; ++it) {
                auto match = *it;
                auto patternOututEdges = match.first->outEdges();
                auto compOutputEdges = match.second->outEdges();

                if (match.first->outNodes().size() != match.second->outNodes().size()) {
                    visitedLastDataNodes.clear();
                    outputApiMatch.clear();
                    matched = false;
                    break;
                }

                for (auto patternIt = patternOututEdges.begin(); patternIt != patternOututEdges.end(); ++patternIt) {

                    if ((*patternIt)->dstNode()->outEdges().size() != 0) {
                        continue;
                    }

                    auto matchedIt = std::find_if(compOutputEdges.begin(), compOutputEdges.end(),
                        [&patternIt, &patternGraph, &compGraph, &visitedLastDataNodes](const ade::EdgeHandle& compEdge) -> bool {
                        auto patternOutputPort = patternGraph.metadata(*patternIt).get<cv::gimpl::Output>().port;
                        auto compOutputPort = compGraph.metadata(compEdge).get<cv::gimpl::Output>().port;

                        if (patternOutputPort != compOutputPort) {
                            return false;
                        }

                        // Get rid of this code
                        // Not sure that it is needed at all, we can't have such case with multiple outputs to 1 data node
                        auto foundit = std::find_if(visitedLastDataNodes.begin(), visitedLastDataNodes.end(), [&compEdge](const ade::NodeHandle& visitedNode) { return compEdge->dstNode() == visitedNode; });
                        if (foundit != visitedLastDataNodes.end()) {
                            return false;
                        }

                        visitedLastDataNodes.insert(compEdge->dstNode());

                        return true;
                    });

                    if (matchedIt == compOutputEdges.end()) {
                        visitedLastDataNodes.clear();
                        outputApiMatch.clear();
                        matched = false;
                        break;
                    }
                    outputApiMatch[(*patternIt)->dstNode()] = (*matchedIt)->dstNode();
                }

            }
        }
    }

    SubgraphMatch subgraph{};

    if (notFound || !matched) {
        return subgraph;
    }

    subgraph.inputDataNodes = inputApiMatch;
    subgraph.startOpNodes = subgraphIns;
    subgraph.internalLayers = subgraphInternals;
    subgraph.finishOpNodes = subgraphOuts;
    subgraph.outputDataNodes = outputApiMatch;

    return subgraph;
}
