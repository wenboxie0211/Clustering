# -*- coding: utf-8 -*-
"""
Created on Fri Apr 07 10:14:29 2017

@author: Administrator
"""
import networkx as nx
from NG import Index


class communityAlg():
    def GN(self, G):
        # source:    algorithm from PRE, 69(2004),026113: Finding and evaluating community structure in networks
        # algorithm: iterative calculate the edge betweenness, and remove the edge corresponding to the largest betweeness
        # remove edge, until the component number changed, compute the modularity until find the first peak
        # end condition: find the first peak of modularity

        currentedgenum = nx.number_of_edges(G)
        intial_com_number = nx.number_connected_components(G)
        current_com_number = intial_com_number
        # max_modularity = 0
        current_modularity = 0
        G1 = G.copy()
        maxG = G.copy()
        idx = Index.Index()
        max_modularity = 0
        # while currentedgenum > 0:
        #     currentedgenum = self.removeAllLargestBetweenessEdges(G1, currentedgenum)
        #     current_com_number = nx.number_connected_components(G1)
        #     if current_com_number > intial_com_number:
        #         intial_com_number = current_com_number
        #
        #         communitySet = nx.connected_components(G1)
        #         current_modularity = idx.modularity_nx(communitySet, G)
        #
        #         if current_modularity < max_modularity:
        #             return nx.connected_components(G1)
        #         else:
        #             max_modularity = current_modularity
        # return nx.connected_components(G1)

        while currentedgenum > 0:
            currentedgenum = self.removeAllLargestBetweenessEdges(G1, currentedgenum)
            current_com_number = nx.number_connected_components(G1)
            if current_com_number > intial_com_number:
                intial_com_number = current_com_number

                communitySet = nx.connected_components(G1)
                current_modularity = idx.modularity_nx(communitySet, G)

                communitySet = nx.connected_components(G1)
                no = 0
                for c in communitySet:
                    no += 1
                print(no, '\t', idx.TriParRatio_nx(nx.connected_components(G1), G))
                # print(current_modularity,';',idx.TriParRatio_nx(nx.connected_components(G1), G),';',idx.conductance_nx(nx.connected_components(G1), G))

                if current_modularity > max_modularity:
                    max_modularity = current_modularity
                    maxG=G1.copy()

        return nx.connected_components(maxG)

    def removeAllLargestBetweenessEdges(self, G, currentedgenum):
        bw = nx.edge_betweenness_centrality(G)
        max_ = 0.0
        for key in bw:
            if bw[key] > max_:
                max_ = bw[key]
        for key in bw:
            if bw[key] == max_:
                G.remove_edge(key[0], key[1])
                currentedgenum -= 1
        return currentedgenum
