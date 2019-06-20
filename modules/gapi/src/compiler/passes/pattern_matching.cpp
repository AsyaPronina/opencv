#include <opencv2/gapi/pattern_matching.hpp>

//first data nodes do not matter for us!!
//only internal connections

cv::gapi::SubgraphMatch cv::gapi::findMatches(cv::gimpl::GModel::Graph patternGraph, cv::gimpl::GModel::Graph compGraph) {
    using Graph = cv::gimpl::GModel::Graph;
    using Metadata = typename Graph::MetadataT;
    using VisitedMatchings = std::list<std::pair<ade::NodeHandle, ade::NodeHandle>>;

    //N^2 check if this graph may exist at all.
    //find last pattern op nodes and stop at this point data graph search.

    std::unordered_set<ade::NodeHandle, cv::gapi::SubgraphMatch::NodeHandleHashFunction> firstPatternOpNodes;
    std::unordered_set<ade::NodeHandle, cv::gapi::SubgraphMatch::NodeHandleHashFunction> lastPatternOpNodes;

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

    auto dataNodesComparator = [](std::pair<const ade::NodeHandle, std::vector<int>> first, Metadata firstMetadata, std::pair<const ade::NodeHandle, std::vector<int>> second, Metadata secondMetadata) {
        if (secondMetadata.get<cv::gimpl::NodeType>().t != cv::gimpl::NodeType::DATA) {
            std::logic_error("NodeType of passed node as second argument shall be NodeType::DATA!");
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

        auto& firstOutputNodes = first.first->outNodes();
        auto& secondOutputNodes = second.first->outNodes();

        if (firstOutputNodes.size() != secondOutputNodes.size()) {
            return false;
        }

        return true;
    };

    auto opNodesComparator = [&matchedVisitedNodes](std::pair<const ade::NodeHandle, std::vector<int>> first, Metadata firstMetadata, std::pair<const ade::NodeHandle, std::vector<int>> second, Metadata secondMetadata, bool& isAlreadyVisited) {
        if (secondMetadata.get<cv::gimpl::NodeType>().t != cv::gimpl::NodeType::OP) {
            //std::logic_error("NodeType of passed node as second argument shall be NodeType::OP!");
            return false;
        }

        // Assuming that if kernels names are the same then output DATA nodes counts from kernels are the same.
        // Assuming that if kernels names are the same then input DATA nodes counts to kernels are the same.
        if (firstMetadata.get<cv::gimpl::Op>().k.name != secondMetadata.get<cv::gimpl::Op>().k.name) {
            return false;
        }

        //Extra for our case, because we can't create graph contained operation, which has multiple returns and all them are located in 1 variable (in 1 DATA node).
        auto& firstOutputNodes = first.first->outNodes();
        auto& secondOutputNodes = second.first->outNodes();

        if (firstOutputNodes.size() != secondOutputNodes.size()) {
            return false;
        } // extra

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

    std::unordered_map<ade::NodeHandle, std::vector<ade::NodeHandle>, cv::gapi::SubgraphMatch::NodeHandleHashFunction> allMatchingsForFirstOpNodes;

    auto compNodes = compGraph.nodes();
    for (auto firstPatternOpNode : firstPatternOpNodes) {
        std::vector<ade::NodeHandle> possibleMatchings;
        std::copy_if(compNodes.begin(), compNodes.end(), std::back_inserter(possibleMatchings), [&firstPatternOpNode, &patternGraph, &compGraph, &opNodesComparator](const ade::NodeHandle& node) {
            auto firstMetadata = patternGraph.metadata(firstPatternOpNode);
            auto secondMetadata = compGraph.metadata(node);
            bool stub = false;
            /* TODO: FIXX */                               return opNodesComparator(std::make_pair(firstPatternOpNode, std::vector<int>{ 0 }), firstMetadata, std::make_pair(node, std::vector<int>{ 0 }), secondMetadata, stub);
        });

        allMatchingsForFirstOpNodes[firstPatternOpNode] = possibleMatchings;
    }

    std::list<std::pair<ade::NodeHandle, ade::NodeHandle>> subgraphIns;
    std::list<std::pair<ade::NodeHandle, ade::NodeHandle>> subgraphOuts;
    std::list<ade::NodeHandle> subgraphInternals;


    // Structural matching first, semantic matching second.
    bool notFound = true;
    int i = 0;
    while (notFound && (i < allMatchingsForFirstOpNodes.size())) {
        subgraphIns.clear();
        subgraphOuts.clear();
        subgraphInternals.clear();
        matchedVisitedNodes.clear();

        int div = i;
        for (auto allMatchingsForFirstOpNode : allMatchingsForFirstOpNodes) { //order is not determined: for ex., for last node. =( use ordered set and map to ensure order. 
            auto size = allMatchingsForFirstOpNode.second.size();
            int index = div % size;
            div = div / size;
            auto firstCompOpNode = allMatchingsForFirstOpNode.second[index];
            matchedVisitedNodes.push_back({ allMatchingsForFirstOpNode.first, firstCompOpNode });
            subgraphIns.push_back(matchedVisitedNodes.back());
        }

        bool nonStop = true;
        bool isSearchFailed = false;

        typename VisitedMatchings::iterator matchIt = matchedVisitedNodes.begin();
        std::size_t size = matchedVisitedNodes.size();
        std::size_t index = 0;

        while (nonStop) {
            for (index; index < size && !isSearchFailed; ++index, ++matchIt) {

                auto foundIt = std::find(lastPatternOpNodes.begin(), lastPatternOpNodes.end(), matchIt->first);
                if (foundIt != lastPatternOpNodes.end()) {
                    subgraphOuts.push_back(*matchIt);
                }
                else {
                    subgraphInternals.push_back(matchIt->second);
                }

                std::unordered_map<ade::NodeHandle, std::vector<int>, cv::gapi::SubgraphMatch::NodeHandleHashFunction> patternOutputNodesLabeled;
                std::unordered_map<ade::NodeHandle, std::vector<int>, cv::gapi::SubgraphMatch::NodeHandleHashFunction> compOutputNodesLabeled;

                auto& patternOutputEdges = matchIt->first->outEdges();
                auto& compOutputEdges = matchIt->second->outEdges();

                auto addLabelToNode = [](ade::NodeHandle node, ade::EdgeHandle edge, const Graph& graph, std::unordered_map<ade::NodeHandle, std::vector<int>, cv::gapi::SubgraphMatch::NodeHandleHashFunction>& labeledNodes) {
                    if (graph.metadata(node).get<cv::gimpl::NodeType>().t == cv::gimpl::NodeType::OP) {
                        labeledNodes[node].push_back(graph.metadata(edge).get<cv::gimpl::Input>().port);
                    }
                    else {
                        labeledNodes[node].push_back(graph.metadata(edge).get<cv::gimpl::Output>().port);
                    }
                };

                for (auto patternOutputEdge : patternOutputEdges) {
                    if (patternOutputEdge->dstNode()->outEdges().size() != 0) {
                        addLabelToNode(patternOutputEdge->dstNode(), patternOutputEdge, patternGraph, patternOutputNodesLabeled);
                    }
                }

                for (auto compOutputEdge : compOutputEdges) {
                    if (compOutputEdge->dstNode()->outEdges().size() != 0) {
                        //Assuming that there is no case for the op node without output data nodes.
                        addLabelToNode(compOutputEdge->dstNode(), compOutputEdge, compGraph, compOutputNodesLabeled);
                    }
                }

                for (auto patternIt = patternOutputNodesLabeled.begin(); patternIt != patternOutputNodesLabeled.end(); ++patternIt) {
                    bool isAlreadyVisited = false;

                    auto matchedIt = std::find_if(compOutputNodesLabeled.begin(), compOutputNodesLabeled.end(),
                        [&patternIt, &patternGraph, &compGraph, &dataNodesComparator, &opNodesComparator, &isAlreadyVisited](std::pair<const ade::NodeHandle, std::vector<int>>& compNode) -> bool {
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

    std::unordered_map<ade::NodeHandle, ade::NodeHandle, cv::gapi::SubgraphMatch::NodeHandleHashFunction> inputApiMatch(firstPatternDataNodes.size());
    std::unordered_map<ade::NodeHandle, ade::NodeHandle, cv::gapi::SubgraphMatch::NodeHandleHashFunction> outputApiMatch(lastPatternDataNodes.size());

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

                    auto foundit = std::find_if(matchedVisitedFirstDataNodes.begin(), matchedVisitedFirstDataNodes.end(), [&patternIt](std::pair<ade::NodeHandle, ade::NodeHandle> match) { return (*patternIt)->srcNode() == match.first; });
                    if (foundit != matchedVisitedFirstDataNodes.end()) {
                        if (compEdge->srcNode() != foundit->second) {
                            return false;
                        }
                    }

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
            std::unordered_set<ade::NodeHandle, cv::gapi::SubgraphMatch::NodeHandleHashFunction> visitedLastDataNodes;
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
                        auto patternInputPort = patternGraph.metadata(*patternIt).get<cv::gimpl::Output>().port;
                        auto compInputPort = compGraph.metadata(compEdge).get<cv::gimpl::Output>().port;

                        if (patternInputPort != compInputPort) {
                            return false;
                        }

                        auto foundit = std::find_if(visitedLastDataNodes.begin(), visitedLastDataNodes.end(), [&patternIt](const ade::NodeHandle& match) { return (*patternIt)->dstNode() == match; });
                        if (foundit != visitedLastDataNodes.end()) {
                            return false;
                        }

                        visitedLastDataNodes.insert(compEdge->srcNode());

                        return true;
                    });

                    if (matchedIt == compOutputEdges.end()) {
                        visitedLastDataNodes.clear();
                        outputApiMatch.clear();
                        matched = false;
                        break;
                    }
                    outputApiMatch[(*patternIt)->srcNode()] = (*matchedIt)->srcNode();
                }

            }
        }
    }

    SubgraphMatch subgraph{};

    if (notFound || !matched) {
        return subgraph;
    }

    subgraph.inputDataNodesMatches = inputApiMatch;
    subgraph.firstOpNodesMatches = subgraphIns;
    subgraph.internalLayers = subgraphInternals;
    subgraph.lastOpNodesMatches = subgraphOuts;
    subgraph.outputDataNodesMatches = outputApiMatch;
    
    return subgraph;
}
