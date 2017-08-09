# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 10:21:28 2017

@author: Administrator
"""
import math
import networkx as nx


class Index:
    def variationInformaiton(self, inputpath):
        # cal the variationInformaion of two clusters partions.
        # input file format: node label1ofMethod label2ofrandom
        clusterSet_method = {}
        clusterSet_random = {}
        nodeset = set()
        # cluster-nodeset格式
        dfile = open(inputpath, 'r')
        dfile.readline()
        for line in dfile:
            line = line.strip('\n')
            dlist = line.split(',')
            nodeset.add(dlist[0])
            if dlist[1] in clusterSet_method:
                clusterSet_method[dlist[1]][0].add(dlist[0])
            else:
                f = set()
                f.add(dlist[0])
                clusterSet_method.setdefault(dlist[1], [f, 0])
            if dlist[2] in clusterSet_random:
                clusterSet_random[dlist[2]][0].add(dlist[0])
            else:
                f = set()
                f.add(dlist[0])
                clusterSet_random.setdefault(dlist[2], [f, 0])
        dfile.close()
        # fenzi = \sumall(i,j) pair |N(i)&N(j)|*log((|N(i)&N(j)|*N)/(N(i)*N(j)))
        # fenmu = \sumall N(i)*log(N(i)/N) + N(i)*log(N(i)/N)
        N = len(nodeset)
        nodeset.clear()

        method_label = list()
        random_label = list()

        fenmu = 0.0
        for i in clusterSet_method:
            method_label.append(i)
            clusterSet_method[i][1] = len(clusterSet_method[i][0])
            fenmu += clusterSet_method[i][1] * math.log(float(clusterSet_method[i][1]) / N, 2)
        for i in clusterSet_random:
            random_label.append(i)
            clusterSet_random[i][1] = len(clusterSet_random[i][0])
            fenmu += clusterSet_random[i][1] * math.log(float(clusterSet_random[i][1]) / N, 2)

        fenzi = 0.0
        for i in method_label:
            for j in random_label:
                intersectNumber = len(clusterSet_method[i][0] & clusterSet_random[j][0])
                if intersectNumber != 0:
                    fenzi += intersectNumber * math.log(
                        float(intersectNumber * N) / (clusterSet_method[i][1] * clusterSet_random[j][1]), 2)
        return (-2 * fenzi) / fenmu

    def modularity_file(self, clusterlabelfile, edgeweight):
        # file version
        ## calculate the modularity of the clusters
        # formula as \sum_{c=1}^{c=n_c}  (w_c/w + (s_c/(2*w))^2)
        # clusterlabelfile format: node clusterlabel
        # edgeweight format: node1,node2, weight

        modular_value = 0.0
        cluster_label = {}
        W = 0.0
        label_value = {}  # format as {label:[w_c,s_c]}
        dfile = open(clusterlabelfile, 'r')
        for line in dfile:
            dlist = line.split('\t')
            cluster_label.setdefault(dlist[0], dlist[1])
            if dlist[1] not in label_value:
                label_value.setdefault(dlist[1], [0.0, 0.0])
        dfile.close()

        dfile = open(edgeweight, 'r')
        for line in dfile:
            line = line.strip('\n')
            dlist = line.split('\t')  # node1,node2,weight

            W += float(dlist[2])
            label_value[cluster_label[dlist[0]]][1] += float(dlist[2])
            label_value[cluster_label[dlist[1]]][1] += float(dlist[2])
            if cluster_label[dlist[0]] == cluster_label[dlist[1]]:
                label_value[cluster_label[dlist[0]]][0] += float(dlist[2])
        dfile.close()

        for key in label_value:
            modular_value += label_value[key][0] / W
            modular_value -= (label_value[key][1] / (2 * W)) ** 2

        return modular_value

    def modularity_nx(self, communitySet, G2):
        # nx Graph version
        # communitySet: nodeset for each community; G2 is the original comprehensive graph corresponding to G1
        label_value = {}  # format as {label:[w_c,s_c]}
        label_index = 0
        W = 0.0
        for comps in communitySet:  # w_c s_c
            s_c = 0.0
            w_c = 0.0
            for node1 in comps:
                neighbors = G2.neighbors(node1)
                for node2 in neighbors:
                    s_c += G2[node1][node2]['weight']
                    if node2 in comps:
                        w_c += G2[node1][node2]['weight']
            w_c = w_c / 2
            W += s_c
            label_value.setdefault(label_index, [w_c, s_c])
            label_index += 1

        modular_value = 0.0
        W = W / 2
        for key in label_value:
            modular_value += label_value[key][0] / W
            modular_value -= (label_value[key][1] / (2 * W)) ** 2

        return modular_value

    def conductance_nx(self, communitySet, G2):
        # nx Graph version
        # communitySet: nodeset for each community; G2 is the original comprehensive graph corresponding to G1
        # formula as \sum_{s=1}^{s=n_s}  (c_s/(2*m_s+c_s))
        # m_s is all the edge weight in the community; equals w_c in modualrity
        # c_s is all the edge weight point out from the community
        # the smaller, the better, no upper bound of conductance

        label_value = {}  # format as {label:[m_s,c_s]}
        label_index = 0
        for comps in communitySet:  # m_s c_s
            c_s = 0.0
            m_s = 0.0

            n_s = 0
            for node1 in comps:
                n_s += 1
                neighbors = G2.neighbors(node1)
                for node2 in neighbors:
                    if node2 in comps:
                        m_s += G2[node1][node2]['weight']
                    else:
                        c_s += G2[node1][node2]['weight']

            m_s = m_s / 2  # one edge have been calculated twice

            label_value.setdefault(label_index, [m_s, c_s])
            label_index += 1

        conductance_value = 0.0

        for key in label_value:
            conductance_value += label_value[key][1] / (2 * label_value[key][0] + label_value[key][1])

        return conductance_value/len(label_value)

    def TriParRatio_nx(self, communitySet, G2):
        # nx Graph version
        # communitySet: nodeset for each community; G2 is the original comprehensive graph corresponding to G1
        # fraction of nodes in the community that belongs to a traid amd all the node of the traid are belongs to the same community
        tri_par_ratio_value = 0.0
        i=0
        for comp in communitySet:
            candidate_nodes = comp.copy()
            in_triad_nodes = set()
            n_s = 0  # node number of communtiy s
            while candidate_nodes:
                node = candidate_nodes.pop()
                n_s += 1

                neighbors = G2.neighbors(node)
                while neighbors:
                    neighbor = neighbors.pop()
                    if neighbor in comp:
                        common_nodes = nx.common_neighbors(G2, node, neighbor)
                        for common_node in common_nodes:
                            if common_node in comp:
                                in_triad_nodes.add(node)
                                in_triad_nodes.add(neighbor)
                                in_triad_nodes.add(common_node)
                                if neighbor in candidate_nodes:
                                    candidate_nodes.remove(neighbor)
                                    n_s += 1
                                if common_node in candidate_nodes:
                                    candidate_nodes.remove(common_node)
                                    n_s += 1

                                if neighbor in neighbors:
                                    neighbors.remove(neighbor)
                                if common_node in neighbors:
                                    neighbors.remove(common_node)

            if n_s != 0:
                tri_par_ratio_value += float(len(in_triad_nodes)) / n_s
                i+=1
        return tri_par_ratio_value/i
