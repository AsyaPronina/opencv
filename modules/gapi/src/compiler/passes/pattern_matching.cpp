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
      //  - DATA node: then vector is 1-element vector containing port number of
      //    the input edge
      //  - OP node: then vector is ports' vector of current connections between
      //    this node and an parent active DATA node
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
        throw std::logic_error("NodeType of passed node as second argument shall be NodeType::DATA!");
    }

    if (firstMetadata.get<cv::gimpl::Data>().shape != secondMetadata.get<cv::gimpl::Data>().shape) {
        return false;
    }

    if (*firstPorts.begin() != *secondPorts.begin()) {
        return false;
    }

    auto firstOutputEdges = first->outEdges();
    auto secondOutputEdges = second->outEdges();

    if (firstOutputEdges.size() != secondOutputEdges.size()) {
        return false;
    }

    // FIXME: Because of new changes which introduce existence of unused DATA nodes
    // check that first and second nodes have the same type of DATA::Storage.

    return true;
};

// Returns true if two OP nodes semantically and structurally identical:
//    - both nodes have the same kernel name
//    - both nodes are produced by the same port numbers
//    - if any of the nodes are in the array with visited matchings, then:
//      first node is equal to found matching first argument and
//      second node is equal to found matching second argument


// @param first - first node to compare
// @param firstPorts - ports' vector of current connections between first node and an parent active DATA node
// @param firstMetadata - metadata of first
// @param second - second node to compare
// @param secondPorts - ports' vector of current connections between second node and an parent active DATA node
// @param secondMetadata - metadata of second
// @param [out] isAlreadyVisited - set to true if first and second nodes have been already visited
bool opNodesComparator(const VisitedMatchings& matchedVisitedNodes,
                       const ade::NodeHandle first, std::vector<std::size_t> firstPorts, Metadata firstMetadata,
                       const ade::NodeHandle second, std::vector<std::size_t> secondPorts, Metadata secondMetadata,
                       bool& isAlreadyVisited) {
    if (secondMetadata.get<cv::gimpl::NodeType>().t != cv::gimpl::NodeType::OP) {
        throw std::logic_error("NodeType of passed node as second argument shall be NodeType::OP!");
    }

    // Assuming that if kernels names are the same then output DATA nodes counts from kernels are the same.
    // Assuming that if kernels names are the same then input DATA nodes counts to kernels are the same.
    if (firstMetadata.get<cv::gimpl::Op>().k.name != secondMetadata.get<cv::gimpl::Op>().k.name) {
        return false;
    }

    std::sort(firstPorts.begin(), firstPorts.end());
    std::sort(secondPorts.begin(), secondPorts.end());
    if (firstPorts != secondPorts) {
        return false;
    }

    // Shall work, but it is good to test on the cases where multiple start pattern OP nodes
    // maps to the test's one.
    auto foundit = std::find_if(matchedVisitedNodes.begin(), matchedVisitedNodes.end(),
                               [&first, &second](std::pair<ade::NodeHandle, ade::NodeHandle> match)
                               {return first == match.first || second == match.second; });
    if (foundit != matchedVisitedNodes.end()) {
        if (first != foundit->first || second != foundit->second) {
            return false;
        }

        isAlreadyVisited = true;
    }

    return true;
};

// Depending on type of the node retrieve port number (IN/OUT) of the edge entering this node.
std::size_t labelOf (ade::NodeHandle node, // reader node
                     ade::EdgeHandle edge, // edge entering the reader node
                     const Graph& graph) { // graph containing node and edge

    if (graph.metadata(node).get<cv::gimpl::NodeType>().t == cv::gimpl::NodeType::OP) {
        return graph.metadata(edge).get<cv::gimpl::Input>().port;
    }
    else {
        return graph.metadata(edge).get<cv::gimpl::Output>().port;
    }
};

inline bool IS_STARTPOINT(const ade::NodeHandle& nh){
    // FIXME: Because of new changes which introduce existence of unused DATA nodes
    // Try to rely on the nh Data::Storage::INPUT
    return nh->inEdges().empty();
}

inline bool IS_ENDPOINT(const ade::NodeHandle& nh){
    // FIXME: Because of new changes which introduce existence of unused DATA nodes
    // Try to rely on the nh Data::Storage::OUTPUT
    return nh->outEdges().empty();
}
}

// Routine relies on the logic that 1 DATA node may have only 1 input edge.
cv::gimpl::SubgraphMatch
cv::gimpl::findMatches(const cv::gimpl::GModel::Graph& patternGraph,
                       const cv::gimpl::GModel::Graph& testGraph) {

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
        // May be switched to lastPatternOpNodes.insert(*opNodes.begin());
        lastPatternOpNodes.insert(opNodes.begin(), opNodes.end());
    }

    std::unordered_map<ade::NodeHandle,              // pattern OP node
                       std::vector<ade::NodeHandle>, // nodes in the test graph which match to the pattern OP node
                       ade::HandleHasher<ade::Node>> allMatchingsForFirstOpNodes;

    //Filling of allMatchingsForFirstOpNodes
    std::size_t possibleStartPointsCount = 1;

    // For every starting OP node of pattern identify matching candidates(there may be many) in test graph.
    auto testNodes = testGraph.nodes();
    for (auto firstPatternOpNode : firstPatternOpNodes) {
        auto firstMetadata = patternGraph.metadata(firstPatternOpNode);

        std::vector<ade::NodeHandle> possibleMatchings;
        std::copy_if(testNodes.begin(), testNodes.end(), std::back_inserter(possibleMatchings),
            [&](const ade::NodeHandle& node) {
            auto secondMetadata = testGraph.metadata(node);

            bool stub = false;
            if (secondMetadata.get<cv::gimpl::NodeType>().t == cv::gimpl::NodeType::OP) {
                return opNodesComparator({ },
                                         firstPatternOpNode, {  }, firstMetadata,
                                         node, {  }, secondMetadata,
                                         stub);
            }
            else {
                return false;
            }
        });

        possibleStartPointsCount *= possibleMatchings.size();
        allMatchingsForFirstOpNodes[firstPatternOpNode] = std::move(possibleMatchings);
    }

    if (possibleStartPointsCount == 0) {
        // Pattern graph is not matched
        return SubgraphMatch { };
    }

    SubgraphMatch::M subgraphStartOps;
    SubgraphMatch::M subgraphEndOps;
    // FIXME: consider moving to S
    std::list<ade::NodeHandle> subgraphInternals;


    // Structural matching first, semantic matching second.

    // 'found' means pattern is matched.
    // FIXME: Consider better naming for 'found'.
    bool found = false;
    std::size_t i = 0;
    while (!found && (i < possibleStartPointsCount)) {
        subgraphStartOps.clear();
        subgraphEndOps.clear();
        subgraphInternals.clear();

        // List of the pairs representing matchings of pattern node to the test node.
        VisitedMatchings matchedVisitedNodes;

        // Cartesian product of candidate sets for start OP nodes gives set of samples as possible matchings for start OP nodes.
        // Let allMatchingsForFirstOpNodes looks like:  x1 : [ y1 ]
        //                                              x2 : [ y2, y3 ]
        // Cartesian product of two these candidates sets (for x1 and x2 pattern nodes correspondingly) produces two samples
        // of matchings for x1, x2:
        //                         [ y1, y2 ]
        //                         [ y1, y3 ]
        //
        // This loop pushes the next sample from the cartesian product of candidates sets to matchedVisitedNodes list.
        std::size_t div = i;
        for (auto allMatchingsForFirstOpNode : allMatchingsForFirstOpNodes) {
            // TODO: order is not determined: for ex., for last node.
            // May be use ordered set and map to ensure order?
            auto size = allMatchingsForFirstOpNode.second.size();

            // i is traversing full cartesian product of candidates sets.
            // The below code block decodes i to a particular combination from that product.
            std::size_t index = div % size;
            div = div / size;
            auto firstTestOpNode = allMatchingsForFirstOpNode.second[index];
            matchedVisitedNodes.push_back({ allMatchingsForFirstOpNode.first, firstTestOpNode });
        }

        bool stop = false;

        // matchIt is an iterator to a pair of pattern ade::NodeHandle to test's ade::nodeHandle.
        auto matchIt = matchedVisitedNodes.begin();
        std::size_t size = matchedVisitedNodes.size();

        while (!stop) {
            // The following loop traverses through the current level of matchings.
            // Every iteration we consider only one certain pair of matched nodes.
            for (std::size_t index = 0u; index < size && !stop; ++index, ++matchIt) {

                // Check if a given matchIt->first node is an pattern-ending OP node.
                // If it is just remember it in a special map.
                bool cond1 = std::find(lastPatternOpNodes.begin(), lastPatternOpNodes.end(), matchIt->first) != lastPatternOpNodes.end();
                if (cond1) {
                    subgraphEndOps[matchIt->first] = matchIt->second;
                }

                // Check if a given matchIt->first node is an pattern-starting OP node.
                // If it is just remember it in a special map.
                bool cond2 = std::find(firstPatternOpNodes.begin(), firstPatternOpNodes.end(), matchIt->first) != firstPatternOpNodes.end();
                if (cond2) {
                    subgraphStartOps[matchIt->first] = matchIt->second;
                }

                // If neither of conditions are true mark the test node as an internal one.
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
                    testOutputNodesLabeled[dstNh].push_back(labelOf(dstNh, testOutputEdge, testGraph));
                }

                //---------------------------------------------------------------------------------
                // Traverse through labeled descendants of pattern node and for every descedant
                // find a matching in labeled descendants of corresponding test node
                for (auto patternNode : patternOutputNodesLabeled) {
                    bool isAlreadyVisited = false;
                    auto patternNodeMetadata = patternGraph.metadata(patternNode.first);

                    auto testIt = std::find_if(testOutputNodesLabeled.begin(), testOutputNodesLabeled.end(),
                        [&](std::pair<const ade::NodeHandle, std::vector<std::size_t>>& testNode) -> bool {
                        auto testNodeMetadata = testGraph.metadata(testNode.first);

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
                        break;
                    }

                    // Update matchedVisitedNodes list with found pair of nodes if the pair
                    // has not been visited before.
                    if (!isAlreadyVisited) {
                        matchedVisitedNodes.push_back({ patternNode.first, testIt->first });
                    }
                } // Loop traversed patternOutputNodesLabeled
            } // Loop traversed matchedVisitedNodes

            // Suppose, pattern and test graphs' structures without input DATA nodes look like:
            //         Pattern graph                Test graph
            //        op1       op2               t_op1      t_op2
            //      +-----+   +-----+            +-----+    +-----+
            //      v     v   v     v            v     v    v     v
            //      d1    d2  d3    d4          t_d1  t_d2 t_d3  t_d4
            //      v     v   v     v            v     v    v     v
            //     ...   ... ...   ...          ...   ...  ...   ...

            // matchedVisitedNodes content before previous loop execution:
            //     op1 <--> t_op1, op2 <--> t_op2
            // matchedVisitedNodes content after previous loop execution (extent with the next level of
            // matchings):
            //     op1 <--> t_op1, op2 <--> t_op2 | d1 <--> t_d1, d2 <---> t_d2, d3 <--> t_d3, d4 <---> t_d4
            //                                           ^
            //                                           |
            //                                      matchIt
            //
            // matchIt iterator points to the first matching in next level if the next level exists.
            // If there is no next level, matchIt == matchedVisitedNodes.end() and all pattern levels
            // (except ones for IN/OUT data nodes) have been already processed, so, pattern subgraph is found.

            if (!stop) {
                // Check if pattetn subgraph is found
                if (matchIt == matchedVisitedNodes.end()) {
                    // Found
                    stop = true;
                    found = true;
                }

                // Update 'size' with the size of the new level of matchings
                size = static_cast<std::size_t>(std::distance(matchIt, matchedVisitedNodes.end()));
            }
        }

        if (!found){
            // Pattern subgraph is not matched.
            // Switch to the next combination of starting points
            ++i;
            continue;
        }
    
        SubgraphMatch::M inputApiMatch;
        SubgraphMatch::M outputApiMatch;
      
        // Traversing current result for starting OPs
        for (auto it = subgraphStartOps.begin(); it != subgraphStartOps.end() && found; ++it) {
            auto match = *it;
            auto patternInputEdges = match.first->inEdges();
            auto testInputEdges = match.second->inEdges();
    
            SubgraphMatch::S patternInNodes(match.first->inNodes().begin(), match.first->inNodes().end());
            SubgraphMatch::S testInNodes(match.second->inNodes().begin(), match.second->inNodes().end());
    
            if (patternInNodes.size() < testInNodes.size()) {
                inputApiMatch.clear();
                found = false;
                break;
            }
            // Else, patternInNodes.size() > testInNodes.size() is considered as valid case.
    
            // Match pattern input DATA nodes with boundary matched test DATA nodes.
            for (auto patternInEdge : patternInputEdges) {
    
                // Not all start OP nodes are located in the beginning of the pattern graph
                // Start OP may have one input DATA node as an Protocol IN node and other
                // input DATA nodes produced from another operations
                if (!IS_STARTPOINT(patternInEdge->srcNode())) {
                    continue;
                }

                auto patternInputPort = patternGraph.metadata(patternInEdge).get<cv::gimpl::Input>().port;

                auto matchedIt = std::find_if(testInputEdges.begin(), testInputEdges.end(),
                    [&](const ade::EdgeHandle& testInEdge) -> bool {
                    auto testInputPort = testGraph.metadata(testInEdge).get<cv::gimpl::Input>().port;
    
                    if (patternInputPort != testInputPort) {
                        return false;
                    }
    
                    auto foundIt = inputApiMatch.find(patternInEdge->srcNode());
                    if (foundIt != inputApiMatch.end()) {
                        if (testInEdge->srcNode() != foundIt->second) {
                            return false;
                        }
                        return true;
                    }

                    // Update inputApiMatch map only if the pair of nodes isn't in the map already
                    inputApiMatch[patternInEdge->srcNode()] = testInEdge->srcNode();
                    return true;
                });
    
                if (matchedIt == testInputEdges.end()) {
                    inputApiMatch.clear();
                    found  = false;
                    break;
                }
            } // Loop traversed patternInputEdges
        } // Loop traversed sugraphStartOps
   
        if (!found) {
            // Pattern IN data nodes can not be matched.
            // Switch to the next combination of starting points
            ++i;
            continue;
        }
 
        // Create vector with the correctly ordered IN data nodes in the test subgraph
        std::vector<ade::NodeHandle> inputTestDataNodes;
        for (auto inPatternNode : firstPatternDataNodes) {
            inputTestDataNodes.push_back(inputApiMatch[inPatternNode]);
        }

        // Traversing current result for ending OPs
        // There is an assumption that if the pattern subgraph is matched, then
        // OUT data nodes shall be definitely matched
        for (auto match : subgraphEndOps) {
            auto patternOutputEdges = match.first->outEdges();
            auto testOutputEdges = match.second->outEdges();
    
            GAPI_Assert(patternOutputEdges.size() == testOutputEdges.size()
                        &&
                        "Ending OP nodes are matched, so OPs' outputs count shall be the same!");

            // Match pattern output DATA nodes with boundary matched test DATA nodes.
            for (auto patternOutEdge : patternOutputEdges) {

                // Not all end OP nodes are located in the ending of the pattern graph
                // End OP node may have one output DATA node as an Protocol OUT node and other
                // output DATA nodes as input for another operations
                if (!IS_ENDPOINT(patternOutEdge->dstNode())) {
                    continue;
                }

                auto patternOutputPort = patternGraph.metadata(patternOutEdge).get<cv::gimpl::Output>().port;

                auto matchedIt = std::find_if(testOutputEdges.begin(), testOutputEdges.end(),
                    [&](const ade::EdgeHandle& testOutEdge) -> bool {
                    auto testOutputPort = testGraph.metadata(testOutEdge).get<cv::gimpl::Output>().port;
    
                    if (patternOutputPort != testOutputPort) {
                        return false;
                    }

                    outputApiMatch[patternOutEdge->dstNode()] = testOutEdge->dstNode();
                    return true;
                });

                GAPI_Assert(matchedIt != testOutputEdges.end()
                            &&
                            "There shall be a match for every OUT data node from ending OP node,"
                            "if ending OP node matches");
            }
    
        }

        // Create vector with the correctly ordered OUT data nodes in the test subgraph
        std::vector<ade::NodeHandle> outputTestDataNodes;
        for (auto outPatternNode : lastPatternDataNodes) {
            outputTestDataNodes.push_back(outputApiMatch[outPatternNode]);
        }
    
        SubgraphMatch subgraph;
    
        subgraph.inputDataNodes = std::move(inputApiMatch);
        subgraph.startOpNodes = std::move(subgraphStartOps);
        subgraph.internalLayers = std::move(subgraphInternals);
        subgraph.finishOpNodes = std::move(subgraphEndOps);
        subgraph.outputDataNodes = std::move(outputApiMatch);
    
        subgraph.inputTestDataNodes = std::move(inputTestDataNodes);
        subgraph.outputTestDataNodes = std::move(outputTestDataNodes);
    
        return subgraph;

    }

    return SubgraphMatch { };
}
