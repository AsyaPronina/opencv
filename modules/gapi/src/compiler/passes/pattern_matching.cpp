// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation

#include <unordered_set>

#include "pattern_matching.hpp"

namespace  {
using Graph = cv::gimpl::GModel::Graph;
using Metadata = typename Graph::CMetadataT;
using VisitedMatchings = std::list<std::pair<ade::NodeHandle, ade::NodeHandle>>;

using L = std::unordered_map
    < // reader node
      ade::NodeHandle
      // if the reader node above is:
      //  - DATA node: is are produced by the same port numbers
      //  - OP node: then vector is ports' vector of current connections between
      //    first node and an parent active DATA node
    , std::vector<std::size_t>
    , ade::HandleHasher<ade::Node>
    >;

// Returns true if two DATA nodes are semantically and structurally identical:
//  - both nodes have the same GShape
//  - both nodes are produced by the same port numbers
//  - both nodes have the same number of output edges
//  (output edges' ports are not checked here)
//
// @param first - first node to compare
// @param firstPorts - a single element vector with first DATA node's producer output port
// @param firstMetadata - metadata of first
// @param second - second node to compare
// @param secondPorts - a single element vector with second DATA node's producer output port
// @param secondMetadata - metadata of second
bool dataNodesComparator(const ade::NodeHandle first, std::vector<std::size_t> firstPorts, Metadata firstMetadata,
                         const ade::NodeHandle second, std::vector<std::size_t> secondPorts, Metadata secondMetadata) {
    if (secondMetadata.get<cv::gimpl::NodeType>().t != cv::gimpl::NodeType::DATA) {
        //TODO: FIX ASAP
        throw std::logic_error("NodeType of passed node as second argument shall be NodeType::DATA!");
    }

    if (firstMetadata.get<cv::gimpl::Data>().shape != secondMetadata.get<cv::gimpl::Data>().shape) {
        return false;
    }

    //todo
    std::sort(firstPorts.begin(), firstPorts.end());
    std::sort(secondPorts.begin(), secondPorts.end());
    if (firstPorts != secondPorts) {
        return false;
    }

    auto firstOutputEdges = first->outEdges();
    auto secondOutputEdges = second->outEdges();

    if (firstOutputEdges.size() != secondOutputEdges.size()) {
        return false;
    }

    return true;
};

// Returns true if two OP nodes semantically and structurally identical:
//    - both nodes have the same kernel name
//    - both nodes are produced by the same port numbers
//    - if first node is visited then check that its pair is equal to second node

// @param first - first node to compare
// @param firstPorts - ports' vector of current connections between first node and an parent active DATA node
// @param firstMetadata - metadata of first
// @param second - second node to compare
// @param secondPorts - ports' vector of current connections between second node and an parent active DATA node
// @param secondMetadata - metadata of second
// @param [out] isAlreadyVisited - set to true if first node was already visited
bool opNodesComparator(const VisitedMatchings& matchedVisitedNodes,
                       const ade::NodeHandle first, std::vector<std::size_t> firstPorts, Metadata firstMetadata,
                       const ade::NodeHandle second, std::vector<std::size_t> secondPorts, Metadata secondMetadata,
                       bool& isAlreadyVisited) {
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

//        // Extra for our case, because we can't create graph contained operation, which has multiple returns and all them are located in 1 variable (in 1 DATA node).
//        auto firstOutputNodes = first->outNodes();
//        auto secondOutputNodes = second->outNodes();

//        if (firstOutputNodes.size() != secondOutputNodes.size()) {
//            return false;
//        }
//        // extra

    //todo
    std::sort(firstPorts.begin(), firstPorts.end());
    std::sort(secondPorts.begin(), secondPorts.end());
    if (firstPorts != secondPorts) {
        return false;
    }
    ;
    //shall be done another way.
    auto foundit = std::find_if(matchedVisitedNodes.begin(), matchedVisitedNodes.end(),
                               [&first](std::pair<ade::NodeHandle, ade::NodeHandle> match)
                               {return first == match.first; });
    if (foundit != matchedVisitedNodes.end()) {
        if (second != foundit->second) {
            return false;
        }

        isAlreadyVisited = true;
    }

    return true;
};

// Depending on type of the node retrieve port number (IN/OUT) of the edge connected To this node.
// Here, "To" means edge, entering this node.
std::size_t labelOf (ade::NodeHandle node, // reader node
                             ade::EdgeHandle edge, // edge leading to reader node
                             const Graph& graph) {   // graph containing node and edge

    if (graph.metadata(node).get<cv::gimpl::NodeType>().t == cv::gimpl::NodeType::OP) {
        return graph.metadata(edge).get<cv::gimpl::Input>().port;
    }
    else {
        //Ruslan: use 1 int instead of vector as for DATA node we can have only 1 input edge.
        return graph.metadata(edge).get<cv::gimpl::Output>().port;
    }
};

inline bool IS_ENDPOINT(const ade::NodeHandle& nh){
    return nh->outEdges().empty();
}
}

cv::gimpl::SubgraphMatch
cv::gimpl::findMatches(const cv::gimpl::GModel::Graph& patternGraph,
                       const cv::gimpl::GModel::Graph& compGraph) {

    //TODO: Possibly, we may add N^2 check whether this graph may match or not at all.
    //      Check that all pattern OP nodes exist in computational graph.

    //---------------------------------------------------------------
    // Identify operations which start and end our pattern
    SubgraphMatch::S firstPatternOpNodes, lastPatternOpNodes;

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


    std::unordered_map<ade::NodeHandle,              // pattern OP node
                       std::vector<ade::NodeHandle>, // test graph matched nodes for the pattern OP
                       ade::HandleHasher<ade::Node>> allMatchingsForFirstOpNodes;

    //Fit the allMatchingsForFirstOpNodes

    std::size_t possibleStartPointsCount = 1;

    // For every starting OP node of pattern identify matching candidates in test graph.
    // For every starting pattern node there may be multiple matching candidates.
    auto testNodes = compGraph.nodes();
    for (auto firstPatternOpNode : firstPatternOpNodes) {
        std::vector<ade::NodeHandle> possibleMatchings;
        std::copy_if(testNodes.begin(), testNodes.end(), std::back_inserter(possibleMatchings),
            [&](const ade::NodeHandle& node) {
            auto firstMetadata = patternGraph.metadata(firstPatternOpNode);
            auto secondMetadata = compGraph.metadata(node);
            bool stub = false;
            /* TODO: FIXX */
            return opNodesComparator(matchedVisitedNodes,
                                     firstPatternOpNode, {  }, firstMetadata,
                                     node, {  }, secondMetadata, stub);
        });

        possibleStartPointsCount *= possibleMatchings.size();
        allMatchingsForFirstOpNodes[firstPatternOpNode] = std::move(possibleMatchings);
    }

    // Bad namings
    // TODO FIX: Use using
    SubgraphMatch::M subgraphStartOps;
    SubgraphMatch::M subgraphEndOps;
    // FIXME: consider moving to S
    std::list<ade::NodeHandle> subgraphInternals;


    // Structural matching first, semantic matching second.

    //TODO: found means pattern is matched except input and output DATA nodes
    bool found = false;
    std::size_t i = 0;
    while (!found && (i < possibleStartPointsCount)) {
        subgraphStartOps.clear();
        subgraphEndOps.clear();
        subgraphInternals.clear();
        matchedVisitedNodes.clear();

        // This loop pushes the next combination from the cartesian product (identified by i, see comment below)
        // to matchedVisitedNodes list.
        // As a result, matchedVisitedNodes contains matching candidates  for the current vector of starting nodes.
        std::size_t div = i;
        for (auto allMatchingsForFirstOpNode : allMatchingsForFirstOpNodes) {
            //order is not determined: for ex., for last node. =( use ordered set and map to ensure order.
            auto size = allMatchingsForFirstOpNode.second.size();

            // i is traversing full cartesian product of every starting OP nodes matches.
            // The below code block decodes i to a particular combination from that space.
            std::size_t index = div % size;
            div = div / size;
            auto firstCompOpNode = allMatchingsForFirstOpNode.second[index];
            matchedVisitedNodes.push_back({ allMatchingsForFirstOpNode.first, firstCompOpNode });
            //subgraphIns.push_back(matchedVisitedNodes.back());
        }

        //think on naming (stop instead)
        bool stop = false;
        bool isSearchFailed = false;

        auto matchIt = matchedVisitedNodes.begin();
        std::size_t size = matchedVisitedNodes.size();
        std::size_t index = 0u;

        while (!stop) {
            for (; index < size && !isSearchFailed; ++index, ++matchIt) {

                // matchIt is a pair of pattern ade::NodeHandle to test's ade::nodeHandle.

                // Check if a given *matchIt node is an ending OP node.
                // If it is just remember it in a special map.
                bool cond1 = std::find(lastPatternOpNodes.begin(), lastPatternOpNodes.end(), matchIt->first) != lastPatternOpNodes.end();
                if (cond1) {
                    subgraphEndOps[matchIt->first] = matchIt->second;
                }

                // Check if a given *matchIt node is an starting OP node.
                // If it is just remember it in a special map.
                bool cond2 = std::find(firstPatternOpNodes.begin(), firstPatternOpNodes.end(), matchIt->first) != firstPatternOpNodes.end();
                if (cond2) {
                    subgraphStartOps[matchIt->first] = matchIt->second;
                }

                // If neither of conditions are true mark this as internal node.
                if (!cond1 && !cond2) {
                    subgraphInternals.push_back(matchIt->second);
                }

                //-------------------------------------------------------------------------------
                // Given the current pattern/test matching of nodes, traverse their descendatnts.
                // For every descendant store the port of the edge connecting to it.
                // NOTE: the nature of port number may vary: it may be either IN for OP nodes
                // or OUT for DATA ones
                L patternOutputNodesLabeled;
                L testOutputNodesLabeled;

                auto patternOutputEdges = matchIt->first->outEdges();
                auto testOutputEdges = matchIt->second->outEdges();

                for (auto patternOutputEdge : patternOutputEdges) {
                    auto dstNh = patternOutputEdge->dstNode();
                    if (!IS_ENDPOINT(dstNh)) {
                        //Assuming that there is no case for the op node without output data nodes.
                        patternOutputNodesLabeled[dstNh].push_back(labelOf(dstNh, patternOutputEdge, patternGraph));
                    }
                }

                for (auto testOutputEdge : testOutputEdges) {
                    auto dstNh = testOutputEdge->dstNode();
                    testOutputNodesLabeled[dstNh].push_back(labelOf(dstNh, testOutputEdge, compGraph));
                }

                //---------------------------------------------------------------------------------
                // Traverse through labeled descendants of pattern node and for every descedant
                // find a matching in labeled descendants of matched test node
                for (auto patternNode : patternOutputNodesLabeled) {
                    bool isAlreadyVisited = false;

                    auto testIt = std::find_if(testOutputNodesLabeled.begin(), testOutputNodesLabeled.end(),
                        [&](std::pair<const ade::NodeHandle, std::vector<std::size_t>>& testNode) -> bool {
                        auto patternNodeMetadata = patternGraph.metadata(patternNode.first);
                        auto testNodeMetadata = compGraph.metadata(testNode.first);

                        if (patternNodeMetadata.get<cv::gimpl::NodeType>().t == cv::gimpl::NodeType::DATA) {
                            return dataNodesComparator(patternNode.first, patternNode.second, patternNodeMetadata,
                                                       testNode.first, testNode.second, testNodeMetadata);
                        }
                        else {
                            return opNodesComparator(matchedVisitedNodes,
                                                     patternNode.first, patternNode.second, patternNodeMetadata,
                                                     testNode.first, testNode.second, testNodeMetadata,
                                                     isAlreadyVisited);
                        }
                    });

                    if (testIt == testOutputNodesLabeled.end()) {
                        stop = true;
                        isSearchFailed = true;
                        break;
                    }

                    //We shall not put in the matchings already visited nodes.
                    if (!isAlreadyVisited) {
                        matchedVisitedNodes.push_back({ patternNode.first, testIt->first });
                    }
                }
            }

            // 2. Secondly, update the matching array
            // Ruslan: no is
            if (!isSearchFailed) {
                if (std::distance(matchIt, matchedVisitedNodes.end()) == 0) {
                    //Found
                    stop = true;
                    found = true;
                }

                index = 0;
                size = static_cast<std::size_t>(std::distance(matchIt, matchedVisitedNodes.end()));
            }
        }

        ++i;
    }

    std::unordered_map<ade::NodeHandle, ade::NodeHandle, ade::HandleHasher<ade::Node>> inputApiMatch(firstPatternDataNodes.size());
    std::unordered_map<ade::NodeHandle, ade::NodeHandle, ade::HandleHasher<ade::Node>> outputApiMatch(lastPatternDataNodes.size());

    bool matched = true;

    if (found) {
        VisitedMatchings matchedVisitedFirstDataNodes;
        for (auto it = subgraphStartOps.begin(); it != subgraphStartOps.end() && matched; ++it) {
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
            for (auto it = subgraphEndOps.begin(); it != subgraphEndOps.end() && matched; ++it) {
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

    if (!found || !matched) {
        return subgraph;
    }

    subgraph.inputDataNodes = inputApiMatch;
    subgraph.startOpNodes = subgraphStartOps;
    subgraph.internalLayers = subgraphInternals;
    subgraph.finishOpNodes = subgraphEndOps;
    subgraph.outputDataNodes = outputApiMatch;

    return subgraph;
}
