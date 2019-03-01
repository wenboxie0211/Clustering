# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 10:19:35 2017

@author: Administrator
"""
import networkx as nx

from NG import CommunityAlg


def readdata(inputpath):
    g = nx.Graph()
    dfile = open(inputpath, 'r')
    for line in dfile:
        dlist = line.split(' ')

        if dlist[0] == 'Source': continue

        # g.add_edge(dlist[0], dlist[1], weight=float(dlist[2]))
        g.add_edge(dlist[0], dlist[1], weight=1.0)
    dfile.close()
    print
    "readover"
    return g


def writedata(communitySet, outputpath):
    dfile = open(outputpath, 'w')
    cluster_index = 0
    dfile.writelines("Id,GN\n")
    for comp in communitySet:
        for node in comp:
            dfile.writelines(str(node) + "," + str(cluster_index) + "\n")
        cluster_index += 1
    print
    "writeover"
    dfile.close()


def main():
    # ==============================================================================
    #     infunc = index()
    #     modular = infunc.modularity_file(clusterlabelfile, weightpath)
    #     G1 = readdata(weightpathremove)
    #     G2 = readdata(weightpath)
    #     modular_nx = infunc.modularity_nx(G1,G2)
    #     vi = infunc.variationInformaiton(clusterlabelfile)
    #     print modular, vi, modular_nx
    # ==============================================================================
    G = readdata(weightpath)
    gn = CommunityAlg.communityAlg()
    communitySet = gn.GN(G)
    writedata(communitySet, clusterlabelfile_result)


if __name__ == "__main__":
    # dataname = 'ucidata-zachary, netsci-379'

   #  clusterlabelfile_result = '/Users/wenboxie/Data/network/'+ dataname +'/ectd.'+ dataname +'-NG.csv'
    name = 'soc-advogato'
    weightpath = '/Users/wenboxie/Data/network/' + name + '/' + name + '-network.txt'
    clusterlabelfile_result = '/Users/wenboxie/Data/network/'+name+'/result-gn.csv'
    main()