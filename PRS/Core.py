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

    # find supporting nodes
    sup_nodes = get_supporting_nodes(R)
    # print('supporting node:', sup_nodes)

    # confirm the tree structure and give labels
    clusters_temp = get_clusters(A, sup_nodes)

    # map to the real id
    index = data._stat_axis.values.tolist()
    clusters = {}
    for sup_node in clusters_temp.keys():
        s1 = index[sup_node[0]]
        s2 = index[sup_node[1]]
        sup_node_real = (s1, s2)
        clusters[sup_node_real] = {}
        for node in clusters_temp[sup_node].keys():
            c_real = index[node]
            p_real = index[clusters_temp[sup_node][node]]
            clusters[sup_node_real][c_real] = p_real
    return clusters

# 确定支撑点
def get_supporting_nodes(R: np.matrix):
    """   
    :param R: 
    :return: 
    """
    candidates = set(range(R.shape[0]))
    supporting_nodes = set()

    for s1 in range(R.shape[0]):
        if R[s1].max() == 2 and s1 in candidates:
            s2 = R[s1].argmax()
            # sup_node = (s1, s2)
            sup_node = s1
            supporting_nodes.add(sup_node)
            candidates.remove(s1)
            candidates.remove(s2)
        else:
            continue
    return supporting_nodes


def get_clusters(A, supporting_nodes):
    cluster = {}
    for sup_node in supporting_nodes:
        # root = create_root(sup_node) # there must be many function to create root #
        cluster[sup_node] = detect_communities(A, sup_node)
    return cluster


def create_root(sup_node):
    return sup_node


def detect_communities(A: np.matrix, parents):
    children = {}
    for p in parents:
        for c in range(A.shape[0]):
            if A[c, p] == 1:
                children[c] = p
    parents_next = children.keys() - parents
    if len(parents_next) != 0:
        children.update(detect_communities(A, parents_next))
    return children

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
    # print(split_threshold)
    for i in range(k - 1):
        sub_data.append(d.iloc[i * split_threshold:(i + 1) * split_threshold, :])
    sub_data.append(d.iloc[(k - 1) * split_threshold:, :])
    return sub_data

