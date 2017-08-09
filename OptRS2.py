from NG import CommunityAlg
import networkx as nx
from NG import Index

dataname = 'netsci-379'

G = nx.Graph()
RS_G = nx.Graph()
betw = nx.Graph()
refile = open('/Users/wenboxie/Data/network/'+dataname+'/result-rs.csv', 'r')
refile.readline()
comm = {}
for line in refile:
    a = line.split(',')
    comm[a[0]]=a[1]
refile.close()
#
efile = open('/Users/wenboxie/Data/network/'+dataname+'/ectd.' + dataname + '.csv', 'r')
efile.readline()
for line in efile:
    dlist = line.split(',')

    s = dlist[0]
    t = dlist[1]
    w = float(dlist[2])
    if comm[s] == comm[t]:
        RS_G.add_edge(dlist[0], dlist[1], weight=w)
    else:
        betw.add_edge(dlist[0], dlist[1], weight=w)
    G.add_edge(dlist[0], dlist[1], weight=w)
efile.close()
#
currentedgenum = nx.number_of_edges(RS_G)
intial_com_number = nx.number_connected_components(RS_G)
current_com_number = intial_com_number


G1 = RS_G.copy()
maxG = RS_G.copy()

idx = Index.Index()
intic=current_modularity = idx.modularity_nx(nx.connected_components(G1), G)
max_modularity = current_modularity
print(current_modularity,';',idx.TriParRatio_nx(nx.connected_components(G1), G),';',idx.conductance_nx(nx.connected_components(G1), G))


#
bw = nx.edge_betweenness_centrality(G)
eo = bw.copy()
ee = nx.edge_betweenness_centrality(betw)
for key in eo:
    if key not in ee.keys():
        bw.pop(key)

#
# while len(bw)>1 and current_modularity <= 0.59:
# while len(bw) > 1 and current_modularity > 0:
while len(bw) > 1:
    min_key = {}
    min_ =1.0
    for key in bw:
        # print(key)
        if bw[key] < min_:
            min_ = bw[key]
            min_key=key
    if comm[min_key[0]] != min_key[1]:
        # print('add:',min_key[0],'->',min_key[1])
        G1.add_edge(min_key[0], min_key[1], weight=1.0)
        communitySet = nx.connected_components(G1)
        current_modularity = idx.modularity_nx(communitySet, G)
        communitySet = nx.connected_components(G1)
        current_TPR = idx.TriParRatio_nx(communitySet, G)
        communitySet = nx.connected_components(G1)
        current_C = idx.conductance_nx(communitySet, G)
        print(current_modularity,';',current_TPR,';',current_C)
    if current_modularity > max_modularity:
        max_modularity = current_modularity
        maxG = G1.copy()
        bw.pop(min_key)
    elif current_modularity == max_modularity:

    else:
        bw = nx.edge_betweenness_centrality(G1)
        eo = bw.copy()
        ee = nx.edge_betweenness_centrality(betw)
        for key in eo:
            if key not in ee.keys():
                bw.pop(key)



#
# while len(bw) > 1 and current_modularity > 0:
#
#     # max_key = {}
#     # max_ =0.0
#     max_key = {}
#     max_ = 0.0
#     for key in bw:
#         # print(key)
#         # print(G.get_edge_data(key[0],key[1])['weight'])
#         if G.get_edge_data(key[0],key[1])['weight'] > max_:
#             max_ = G.get_edge_data(key[0],key[1])['weight']
#             max_key=key
#     if comm[max_key[0]] != max_key[1]:
#         G1.add_edge(max_key[0], max_key[1], weight=1.0)
#     communitySet = nx.connected_components(G1)
#     current_modularity = idx.modularity_nx(communitySet, G)
#     communitySet = nx.connected_components(G1)
#     current_TPR = idx.TriParRatio_nx(communitySet, G)
#     communitySet = nx.connected_components(G1)
#     current_C = idx.conductance_nx(communitySet, G)
#     print(current_modularity,';',current_TPR,';',current_C)
#     if current_modularity > max_modularity:
#         max_modularity = current_modularity
#         maxG = G1.copy()
#     bw.pop(key)

#
print(max_modularity)
communitySet=nx.connected_components(maxG)


#
# dfile = open('/Users/wenboxie/Data/network/'+dataname+'/label-rs-opt.csv', 'w')
# dfile.writelines('Id,rs-opt\n')
# cluster_index = 0
# for comp in communitySet:
#     for node in comp:
#         dfile.writelines(str(node) + "," + str(cluster_index) + "\n")
#     cluster_index += 1
# print
# "writeover"
# dfile.close()