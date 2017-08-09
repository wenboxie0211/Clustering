from NG import CommunityAlg
import networkx as nx
from NG import Index

#netsci-379;beach;football; LFR-1000
dataname = 'football'

G = nx.Graph()
RS_G = nx.Graph()
betw = nx.Graph()
refile = open('/Users/wenboxie/Data/network/'+dataname+'/result-rs5.csv', 'r')
refile.readline()
comm = {}
for line in refile:
    a = line.split(',')
    comm[a[0]]=a[1]
refile.close()
#
efile = open('/Users/wenboxie/Data/network/'+dataname+'/ectd.csv', 'r')
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
        RS_G.add_node(dlist[0])
        RS_G.add_node(dlist[1])
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
# print('rs:',current_modularity,';',idx.TriParRatio_nx(nx.connected_components(G1), G),';',idx.conductance_nx(nx.connected_components(G1), G))
communitySet = nx.connected_components(G1)
no = 0
for c in communitySet:
    no += 1
print(no,'\t', idx.TriParRatio_nx(nx.connected_components(G1), G))

max_TPR =idx.TriParRatio_nx(nx.connected_components(G1), G)

#
bw = nx.edge_betweenness_centrality(G)
eo = bw.copy()
ee = nx.edge_betweenness_centrality(betw)

#write betweeness
out_betweeness = open('/Users/wenboxie/Data/network/'+ dataname +'/betweeness.'+dataname+'.csv', 'w')
out_betweeness.write('Source,Target,Weight\n')

for key in eo:

    #w
    out_betweeness.write(str(key[0]) + ',' + str(key[1]) + ',' + str(bw[key]) + '\n')

    if key not in ee.keys():
        bw.pop(key)
out_betweeness.close()
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
    if comm[min_key[0]] != comm[min_key[1]]:
        # print('add:',min_key[0],'(',comm[min_key[0]],')->',min_key[1],'(',comm[min_key[1]],'')

        G1.add_edge(min_key[0], min_key[1], weight=1.0)

        communitySet = nx.connected_components(G1)
        current_modularity = idx.modularity_nx(communitySet, G)
        communitySet = nx.connected_components(G1)
        current_TPR = idx.TriParRatio_nx(communitySet, G)
        communitySet = nx.connected_components(G1)
        current_C = idx.conductance_nx(communitySet, G)

        communitySet = nx.connected_components(G1)
        no = 0
        for c in communitySet:
            no += 1
        print(no,'\t',current_TPR)

    if current_modularity > max_modularity:
        max_modularity = current_modularity
        maxG = G1.copy()
    # if current_TPR > max_TPR:
    #     max_TPR = current_TPR
    #     maxG = G1.copy()
    bw.pop(min_key)
    # print('G1:',len(G1.nodes()))
    # print('MAX:',len(maxG.nodes()))
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

communitySet = nx.connected_components(maxG)
no = 0
for c in communitySet:
    no += 1

print('MAX:',no,'\t',max_TPR)

communitySet = nx.connected_components(maxG)


dfile = open('/Users/wenboxie/Data/network/'+dataname+'/result-rs           -opt-m.csv', 'w')
dfile.writelines('Id,rs-opt-m\n')
cluster_index = 0
for comp in communitySet:
    for node in comp:
        dfile.writelines(str(node) + "," + str(cluster_index) + "\n")
    cluster_index += 1
print
"writeover"
dfile.close()