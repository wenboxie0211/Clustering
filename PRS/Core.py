import random

import numpy as np
from sklearn.neighbors import NearestNeighbors


# 节点
class node:
    name = -1
    # cluster_number = None
    parent = -1
    children = set()

    def __init__(self, name):
        self.name = name


class tree:
    """
    
    """
    root = None

    def __init__(self, root):
        self.root = root

    def traverse(self):

    def get_nodes(self):

    def get_skeleton(self):



# 原始数据
# 应该是全局存储的，这样可以拓展到分布式算法中




# 聚合
def aggregation(leaves: set(node), data: []):
    """
    
    :param leaves: the set of aggregation candidates
    :param data: global variable
    :return: the set of roots
    """

    # leaves' map & sub_data
    leaves_map = {};
    D = []
    for l, i in leaves, range(len(leaves)):
        leaves_map[i] = l.name
        D.append(data[l.name])

    # adjacent matrix
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(D)
    A = neigh.kneighbors_graph(D)

    # relational matrix
    R = A + A.T

    # find supporting nodes
    sup_nodes = get_supporting_nodes(R, leaves_map)

    # confirm the tree structure and give labels
    clusters = get_clusters(R, sup_nodes)

    return clusters, R

# 确定支撑点
def get_supporting_nodes(R):
    """
    
    :param R: 
    :return: 
    """
    candidates = set(range(len(R)))
    supporting_nodes = set()
    while len(candidates) > 0:
        node_checking = candidates.pop()
        if max(R[node_checking]) != 2:
            candidates.remove(node_checking)
            continue
        else:
            start_node = set(node_checking)
            nns = get_supporting_nodes_checking(R, set(start_node))

        pro = np.array([len(nns)][2])
        for n,i in nns,range(len(nns)):
            candidates.remove(n)
            pro[i][0] = n
            pro[i][1] = sum(R[n])
            if i != 0:
                pro[i][1] += pro[i-1][1]

        r = random.randint(0,pro[-1][1])
        for i in range(len(pro)):
            if r <= pro[i][1]:
                supporting_nodes.add(i)
    return supporting_nodes

# 得到待选支撑点所构成的联通分支
def get_supporting_nodes_checking(R, nns):
    nns_new = set()
    for node in nns:
        for n in range(len(R[node])):
            if R[node][n] == 2 and n not in nns:
                nns_new.add(n)
    if len(nns_new) != 0:
        nns.add(get_supporting_nodes_checking(R,nns_new))

    return nns

# 根据支撑点确定簇内节点
def get_clusters(R, sup_nodes):
    clusters = {}
    for sn in sup_nodes:
        clusters[sn] = get_community_nodes(R, set(sn))
    return clusters

# 划分联通分支
def get_community_nodes(R, nns):
    nns_new = set()
    for node in nns:
        for n in range(len(R[node])):
            if R[node][n] != 0 and n not in nns:
                nns_new.add(n)
    if len(nns_new) != 0:
        nns.add(get_community_nodes(R, nns_new))

    return nns

# 欧式距离
def get_eu_dis(a,b):
    return np.linalg.norm(a-b)

# 数据划分
def data_partition(data, **kwargs):
    sub_data = []
    data_index = []
    if 'num_sub' in kwargs.keys():
        k = kwargs['num_sub']
    else:
        k = 1
    for i in range(0,k):
        sub_data.append([])
        data_index.append([])
    for i in range(0,len(data)):
        sub_data[i%k].append(data[i])
        data_index[i%k].append(i)

    return sub_data

# # 根节点反馈
# def supported_nodes_feedback(old_supported_nodes, old_labels):
#
#     upper_supported_nodes, upper_labels = aggregation(old_supported_nodes)
#     label_map = {}
#     for o_n in old_supported_nodes:
#         old_labels[old_supported_nodes]
#
#
# # 查询节点的追随者
# def supported_node_followers(s):
#     children = s.children
#     if children is None:
#         return s
#     else:
#         followers = set()
#         for c in children:
#             followers.add(supported_node_followers(c))
#         return followers

