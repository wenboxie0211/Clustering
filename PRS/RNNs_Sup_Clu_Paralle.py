import random
import threading
import time

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

    def divide_data_random(self, k):
        d = self.data.take(np.random.permutation(data.shape[0]))
        split_threshold = int(d.shape[0] / k)
        sub_data = [d.iloc[i * split_threshold:(i + 1) * split_threshold, :] for i in range(k - 1)]
        sub_data.append(d.iloc[(k - 1) * split_threshold:, :])
        return sub_data

    def divide_data_results(self, results):
        sub_data = []
        results_map_no = {}
        sup = set(results.values())
        for key, i in zip(sup, range(len(sup))):
            results_map_no[key] = i
            sub_data.append(pd.DataFrame())

        for n in results.keys():
            sub_data[results_map_no[results[n]]] = sub_data[results_map_no[results[n]]].append(self.data.loc[n, :])

        return sub_data

    def get_clusters(self, num_thread, threshold_clusters):
        sub_data = self.divide_data_random(num_thread)
        self.results = self.iteraction(sub_data, threshold_clusters)

        # for i in range(2):
        sub_data = self.divide_data_results(self.results)
        self.results = self.iteraction(sub_data, threshold_clusters / 2)

    def iteraction(self, sub_data, threshold_clusters):
        sub_clusters = []
        # print('length of sub-cluster:', len(sub_data))
        thr = [cluster(i, sub_data[i], threshold_clusters, sub_clusters) for i in range(len(sub_data))]
        for t in thr: t.start()
        for t in thr: t.join()
        # print(len(sub_clusters))
        edges = self.reduce(sub_clusters, threshold_clusters)
        # {parent: set(children)}
        clustering_tree = {}
        roots = set()
        for l in edges:
            if l[0] == l[1]: roots.add(l[1])

            if l[1] not in clustering_tree.keys():
                nc = set()
                nc.add(l[0])
                clustering_tree[l[1]] = nc
            else:
                nc = clustering_tree[l[1]]
                nc.add(l[0])

        # {node: label}
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

    def reduce(self, sub_clusters: [], threshold_clusters):
        data_roots = []
        results = []
        # print('sub_clusters:', sub_clusters)
        while len(sub_clusters) > 0:
            n = sub_clusters.pop()
            # n_0 is the child, n_1 is the parent
            if n[0] == n[1]:
                data_roots.append(self.data.loc[n[0], :])
            else:
                results.append(n)
        # data_root = self.data[sub_clusters[:,0] == sub_clusters[:,0]]
        data_roots = pd.DataFrame(data_roots)
        # print('reduce:',data_roots)
        thread = cluster(999, data_roots, threshold_clusters, results)
        thread.start()
        thread.join()
        return results

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


class cluster(threading.Thread):
    def __init__(self, threadID, data, threshold_clusters, sub_clusters):
        threading.Thread.__init__(self)
        self.data = data
        self.threadID = threadID
        self.threshold_clusters = threshold_clusters
        self.sub_clusters = sub_clusters

    def run(self):
        self.sub_clusters = self.sub_clusters.extend(self.aggregate(self.data))

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
        results = [[row_names[i], row_names[A[i].argmax()]] for i in range(A.shape[0]) if row_names[i] not in sup_nodes]

        # 5. if the number of roots bigger than k, we regard the roots as nodes to aggregate
        if len(sup_nodes) > self.threshold_clusters:
            data_roots = data[data.index.isin(sup_nodes)]
            # print('5:', data_roots)
            # print("thread-", self.threadID, " data_roots: ", data_roots)
            results.extend(self.aggregate(data_roots))
        # 6. otherwise, in A, roots direct to;
        else:
            results.extend([[i, i] for i in sup_nodes])

        return results

    def get_supporting_nodes(self, R, row_names):
        candidates = set(range(R.shape[0]))
        supporting_nodes = set()
        for s1 in range(R.shape[0]):
            if R[s1].max() == 2 and s1 in candidates:
                supporting_nodes.add(row_names[s1])
                candidates.remove(R[s1].argmax())
                candidates.remove(s1)
            else:
                continue
        return supporting_nodes


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
    file_name = '/Users/wenboxie/Data/uci-20070111/exp/iris.txt'
    data = pd.read_csv(file_name, header=None).iloc[:, 0:-1]
    label = pd.read_csv(file_name, header=None).iloc[:, -1]
    # print(label)
    prs = PRS(data)
    start = time.time()
    prs.get_clusters(2, 10)
    # draw_matrix(prs.results)
    # print(prs.results)
    r = prs.get_results()
    # print(r)
    print('k =', len(set(r.values())))
    end = time.time()
    print(end - start)
    r = np.array(sorted(r.items(), key=lambda item: item[0]))[:, 1]
    # print(r)
    print('waite for estimation!')

    ri = metrics.cluster.adjusted_rand_score(label, r)
    # ri = estimate.rand_index(label, r)
    print('ri =', ri)
    start = time.time()
    kmeans_model = cluster_methods.KMeans(n_clusters=3, random_state=1, init='random').fit(data)
    print('ri_kmeans =', metrics.cluster.adjusted_rand_score(label, kmeans_model.labels_))
    end = time.time()
    # print(end - start)
    start = time.time()
    agg_model = cluster_methods.AgglomerativeClustering(n_clusters=3).fit(data)
    print('ri_GA =', metrics.cluster.adjusted_rand_score(label, agg_model.labels_))

    end = time.time()
    # print(end - start)

    # 按行重排列
    # print(data.iloc[:10,:].take(np.random.permutation(10)))
    #
    # file = open(file_name, 'r')
    # for l in file.readlines():
    #     data.append(l.strip('\n').split(',')[0:4])

    # iterating(data, 2)
