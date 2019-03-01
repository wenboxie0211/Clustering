import random

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


#
# # 节点
# class node:
#     name = -1
#     # cluster_number = None
#     parent = -1
#     children = set()
#
#     def __init__(self, name):
#         self.name = name
#
#
# class tree:
#     """
#
#     """
#     root = None
#
#     def __init__(self, root):
#         self.root = root
#
#     def traverse(self):
#
#     def get_nodes(self):
#
#     def get_skeleton(self):



# 原始数据
# 应该是全局存储的，这样可以拓展到分布式算法中




# 聚合
def aggregation(data: pd.DataFrame):
    """
    
    :param leaves: the set of aggregation candidates
    :param data: global variable
    :return: the set of roots
    """

    # # leaves' map & sub_data
    # leaves_map = {};
    # D = []
    # for l, i in leaves, range(len(leaves)):
    #     leaves_map[i] = l.name
    #     D.append(data[l.name])

    # disturb the data
    d = data.values
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            d[i, j] += 0.00001 * random.random()
    # print(d)

    # adjacent matrix
    neighbors = NearestNeighbors(n_neighbors=2)
    neighbors.fit(d)
    A = neighbors.kneighbors_graph(d) - np.eye(len(d))

    # relational matrix
    R = A + A.T
    # print(R)
    # find supporting nodes
    sup_nodes = get_supporting_nodes(R)
    print(sup_nodes)
    # confirm the tree structure and give labels
    clusters_temp = get_clusters(R, sup_nodes)

    # map to the real id
    index = data._stat_axis.values.tolist()
    clusters = {}
    for sup_node in clusters_temp.keys():
        clusters[index[sup_node]] = set()
        for node in clusters_temp[sup_node]:
            clusters[index[sup_node]].add(index[node])

    return clusters

# 确定支撑点
def get_supporting_nodes(R: np.matrix):
    """
    
    :param R: 
    :return: 
    """
    candidates = set(range(R.shape[0]))
    supporting_nodes = set()
    while len(candidates) > 0:
        node_checking = candidates.pop()
        if R[node_checking].max() != 2:
            continue
        else:
            start__ = set()
            start__.add(R[node_checking].argmax())
            nns = get_supporting_nodes_checking(R, start__)

        # print(nns)
        pro = np.zeros([len(nns), 2])
        for n, i in zip(nns, range(len(nns))):
            if n != node_checking:
                candidates.remove(n)
            # print(i)
            pro[i][0] = n
            pro[i][1] = R[n].sum()
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
        if R[node].max() == 2:
            # new_node = set()
            # new_node.add(R[node].argmax())
            nns_new.add(R[node].argmax())
    nns_new = nns_new - nns
    if len(nns_new) != 0:
        nns.update(nns_new)
        nns.update(get_supporting_nodes_checking(R, nns))

    return nns

# 根据支撑点确定簇内节点
def get_clusters(R, sup_nodes):
    clusters = {}
    for sn in sup_nodes:
        sn_set = set()
        sn_set.add(sn)
        clusters[sn] = get_community_nodes(R, sn_set)
    return clusters

# 划分联通分支
def get_community_nodes(R, nns):
    nns_new = set()
    for node in nns:
        for n in range(len(R[node])):
            if R[node, n] != 0 and n not in nns:
                nns_new.add(n)
    if len(nns_new) != 0:
        nns.update(get_community_nodes(R, nns_new))

    return nns

# 欧式距离
def get_eu_dis(a,b):
    return np.linalg.norm(a-b)

# 数据划分
def data_partition(data, **kwargs):
    d = data.take(np.random.permutation(data.shape[0]))
    sub_data = []
    if 'num_sub' in kwargs.keys():
        k = kwargs['num_sub']
    else:
        k = 2

    split_threshold = int(d.shape[0] / k)
    print(split_threshold)
    for i in range(k - 1):
        sub_data.append(d.iloc[i * split_threshold:(i + 1) * split_threshold, :])
    sub_data.append(d.iloc[(k - 1) * split_threshold:, :])
    return sub_data

