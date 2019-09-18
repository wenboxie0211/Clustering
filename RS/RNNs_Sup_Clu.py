"""
实验3，尝试做一个确定根结点的方法
实验4，让结果叠加起来，形成一个带权网络

"""
import math
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import cluster as cluster_methods
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors


class PRS():
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.clusters = []
        self.times = 0

    def get_clusters(self, iteration_times):
        self.iteration_times = iteration_times

        clustering_tree, roots = get_tree(self.aggregate(self.data))

        self.results = get_result(roots, clustering_tree)

    def aggregate(self, data: pd.DataFrame):
        # 1. disturb the data (to make sure that a cluster tree only has one couple of reciprocal nearest neighbor)
        d, row_names = disturb_data(data)
        # print('1:')
        # print(data)
        # print(row_names)
        # 2. get the adjacent matrix and the corresponding relational matrix
        A, R = get_adjacent_matrix(d)

        # 3. get supporting nodes
        sup_nodes = self.get_supporting_nodes(R, row_names)
        # print("thread-", self.threadID, '3:supporting node:', sup_nodes)

        # 4.  对于不是根结点的节点，看作已经确定了邻居，此时就返回其指向，对于根结点就看作没有确定的节点，近一步探索，迭代到下一层。
        # results = [[row_names[i], row_names[A[i].argmax()]] for i in range(A.shape[0]) if row_names[i] not in sup_nodes]
        edges = {}
        for i in range(A.shape[0]):
            if row_names[i] not in sup_nodes:
                edges[row_names[i]] = row_names[A[i].argmax()]

        # 5. pruning
        """
        这部分可以在后续做一下实验，可否去除，出去以后的影响
        """
        # edges, sup_nodes = self.pruning(edges, sup_nodes)

        # 6. if the number of roots higher than k, we regard the roots as nodes to aggregate
        self.times += 1
        if self.times < self.iteration_times:
            data_roots = data[data.index.isin(sup_nodes)]
            # print('5:', data_roots)
            # print("thread-", self.threadID, " data_roots: ", data_roots)
            edges.update(self.aggregate(data_roots))
        # 7. otherwise, in A, roots direct to;
        else:
            for i in sup_nodes: edges[i] = i
            # print('root:', i)

        return edges

    def get_results(self):
        return self.results

    def detect_communities(self, A: np.matrix, roots, labels):
        parents_next = roots
        while len(parents_next) != 0:
            labels_new = set()
            for p in parents_next:
                # print('p=',p)
                for c in range(A.shape[0]):
                    if A[c, p] == 1:
                        labels[c] = labels[p]
                        # print(c, '->', labels[p],'(',c,'->',p,')')
                        labels_new.add(c)
            if len(labels_new) > 0:
                parents_next = labels_new - parents_next
            else:
                break

    def pruning(self, edges, sup_nodes):
        tree = get_tree(edges)[0]
        # print('pruning-sup_node:', sup_nodes)
        # print('tree', tree)
        for root in sup_nodes:
            # 计算高度的阈值
            h = get_height_th_tree(tree, root)
            # print('h=',h)
            ps = set([root])
            for i in range(h - 1):
                ps_new = set()
                for p in ps:
                    if len(tree[p]) > 0:
                        ps_new.union(tree[p])
                ps = ps_new

            while len(ps) > 0:
                ps_new = set()
                for p in ps:
                    # print('h=',h,'p=', p)
                    edges[p] = p
                    sup_nodes.add(p)
                    if len(tree[p]) > 0:
                        ps_new.union(tree[p])
                ps = ps_new

        # print(sup_nodes)
        return edges, sup_nodes

    def get_supporting_nodes(self, R, row_names):

        """
        这部分是随机的去取
        """
        # candidates = set(range(R.shape[0]))
        # supporting_nodes = set()
        # for s1 in range(R.shape[0]):
        #     if R[s1].max() == 2 and s1 in candidates:
        #         supporting_nodes.add(row_names[s1])
        #         candidates.remove(R[s1].argmax())
        #         candidates.remove(s1)
        #     else:
        #         continue
        # return supporting_nodes

        """
            这部分是有策略的去取
        """
        candidates = set(range(R.shape[0]))
        supporting_nodes = set()
        for s1 in range(R.shape[0]):
            if R[s1].max() == 2 and s1 in candidates:
                # s1和s2 都是supporting node
                s2 = R[s1].argmax()

                # 分析哪一个更适合当作为根结点
                ## 1.基于度的比较(度大的作为根结点)

                degree_1 = R[s1].sum()
                degree_2 = R[s2].sum()
                if degree_1 >= degree_2:
                    supporting_nodes.add(row_names[s1])
                else:
                    supporting_nodes.add(row_names[s2])
                candidates.remove(s2)
                candidates.remove(s1)
            else:
                continue
        return supporting_nodes


def get_tree(edges):
    clustering_tree = {}
    roots = set()
    for c, p in edges.items():

        if c == p:
            # print(c, ':', p)
            roots.add(p)

        if p not in clustering_tree.keys():
            nc = set()
            nc.add(c)
            clustering_tree[p] = nc
        else:
            nc = clustering_tree[p]
            nc.add(c)
    # print(roots)
    return clustering_tree, roots


def get_height_th_tree(tree, root):
    n = 1
    ps = set([root])
    while len(ps) > 0:
        ps_new = set()
        for p in ps:
            if len(tree[p]) > 0:
                ps_new.union(tree[p])
                n += len(tree[p])
        ps = ps_new
    # print('n=',n)
    return math.ceil(math.log2(n))
    # return 5


def get_result(roots, clustering_tree):
    result = {}
    for r in roots: result[r] = r
    # print('roots:',roots)
    parents_next = roots
    while len(parents_next) != 0:
        labels_new = set()
        for p in parents_next:
            if p in clustering_tree.keys():
                for c in clustering_tree[p]:
                    result[c] = result[p]
                    # print(c, '->', labels[p],'(',c,'->
                    # ,p,')')
                labels_new = clustering_tree[p] | labels_new
        if len(labels_new) > 0:
            parents_next = labels_new - parents_next
        else:
            break
    return result


def disturb_data(data):
    d = data.values
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            d[i, j] += 0.00001 * random.random()
    # get the name of rows
    return d, data._stat_axis.values.tolist()


def get_adjacent_matrix(d):
    neighbors = NearestNeighbors(n_neighbors=2)
    neighbors.fit(d)
    A = neighbors.kneighbors_graph(d) - np.eye(len(d))
    R = A + A.T
    return A, R


def adjacent_list_2_children_map(results):
    re_map = {}
    for p in results:
        re_map[p[0]] = p[1]
    print(len(results), ',', len(re_map))


def get_midpoint(sup_node):
    midpoint = (data.values[sup_node[0]] + data.values[sup_node[1]]) / 2
    return midpoint


def get_labels(clusters, data_size):
    labels = -1 * np.ones(data_size)
    for i, root in zip(range(len(clusters.keys())), clusters.keys()):
        for node in clusters[root].keys():
            labels[node] = i
    return labels


def draw_matrix(m):
    data = np.array(m)
    # print(data[:,0])
    plt.scatter(data[:, 0], data[:, 1])
    plt.show()


if __name__ == '__main__':
    # {"breast-w", "ecoli", "glass", "ionosphere", "iris", "kdd_synthetic_control", "mfeat-fourier", "mfeat-karhunen","mfeat-zernike"};
    # {"optdigits", "segment", "sonar", "vehicle", "waveform-5000", "letter", "kdd_synthetic_control"};
    file_name = '/Users/wenboxie/Data/uci-20070111/exp/mfeat-fourier.txt'
    data = pd.read_csv(file_name, header=None).iloc[:, 0:-1]
    # data = (data - data.min()+0.001) / (data.max() - data.min()+0.001)
    label = pd.read_csv(file_name, header=None).iloc[:, -1]
    # print(label)
    # prs = PRS(data)
    # start = time.time()
    # prs.get_clusters(2, 10)
    # # draw_matrix(prs.results)
    # # print(prs.results)
    # r = prs.get_results()
    # # print(r)
    # print('k =', len(set(r.values())))
    # end = time.time()
    # # print('time:',end-start)
    # r = np.array(sorted(r.items(),key=lambda  item:item[0]))[:, 1]
    # # print(r)
    # print('waite for estimation!')
    #
    # ri = metrics.cluster.adjusted_rand_score(label, r)


    #############


    for i in range(1, 10):
        prs = PRS(data)
        prs.get_clusters(i)
        r = prs.get_results()
        # print('k =', len(set(r.values())))
        RS_labels_ = np.array(sorted(r.items(), key=lambda item: item[0]))[:, 1]
        if len(set(r.values())) == 1: break
        print('ri_RS(k=', len(set(r.values())), ') =', metrics.cluster.adjusted_rand_score(label, RS_labels_))

        # f = open('/Users/wenboxie/Data/PRS_201904/1_labels.csv', 'w')
        # f.write('id,label\n')
        # for i in range(len(label)):
        #     f.write(str(i)+','+ str(label[i]) + '\n')
        # f.close()

    k = 10

    kmeans_model = cluster_methods.KMeans(n_clusters=k, random_state=1, init='random').fit(data)
    print('ri_kmeans =', metrics.cluster.adjusted_rand_score(label, kmeans_model.labels_))

    agg_model = cluster_methods.AgglomerativeClustering(n_clusters=k).fit(data)
    print('ri_GA =', metrics.cluster.adjusted_rand_score(label, agg_model.labels_))

    ap_model = cluster_methods.AffinityPropagation().fit(data)
    print('ri_ap =', metrics.cluster.adjusted_rand_score(label, ap_model.labels_))

    birch_model = cluster_methods.Birch(n_clusters=k).fit(data)
    print('ri_birch =', metrics.cluster.adjusted_rand_score(label, birch_model.labels_))




    # print(end - start)

    # 按行重排列
    # print(data.iloc[:10,:].take(np.random.permutation(10)))
    #
    # file = open(file_name, 'r')
    # for l in file.readlines():
    #     data.append(l.strip('\n').split(',')[0:4])

    # iterating(data, 2)
