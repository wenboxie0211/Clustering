"""
实验3，尝试做一个确定根结点的方法
实验4，让结果叠加起来，形成一个带权网络

"""
import math
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
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

        # for k in clustering_tree.keys():
        #     for v in clustering_tree[k]:
        #         print(self.data.values[v,0],'\t', self.data.values[v,1],'\t',self.data.values[k,0],'\t', self.data.values[k,1])
        self.results = get_result(roots, clustering_tree)

    def aggregate(self, data: pd.DataFrame):
        # 1. disturb the data (to make sure that a cluster tree only has one couple of reciprocal nearest neighbor)
        row_names = data._stat_axis.values.tolist()
        # d, row_names = disturb_data(data)
        # print('1:')
        # print(data)
        # print(row_names)
        # 2. get the adjacent matrix and the corresponding relational matrix
        A, R = get_adjacent_matrix(data)

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
        # """
        # 这部分可以在后续做一下实验，可否去除，出去以后的影响
        # """
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

    def pruning(self, edges, sup_nodes):
        tree = get_tree(edges)[0]
        # print('pruning-sup_node:', sup_nodes)
        # print('tree:', tree)
        # print('roots:', sup_nodes)

        # 对每一个最小生成树碎片进行剪枝
        cut_nodes = set()
        for root in sup_nodes:
            # 计算高度的阈值
            h = get_height_th_tree(tree, root)
            # print('root:',root,';','h=',h)
            ps = set([root])
            for i in range(h + 1):
                ps_new = set()
                for p in ps:
                    if p in tree.keys() and len(tree[p]) > 0:
                        ps_new = ps_new.union(tree[p])
                ps = ps_new

            while len(ps) > 0:
                ps_new = set()
                for p in ps:
                    # print('h=',h,'p=', p)
                    edges[p] = p
                    cut_nodes.add(p)
                    if p in tree.keys() and len(tree[p]) > 0:
                        ps_new = ps_new.union(tree[p])
                ps = ps_new

        sup_nodes = sup_nodes.union(cut_nodes)
        # print(sup_nodes)
        return edges, sup_nodes

    def get_supporting_nodes(self, R, row_names):

        # 随机
        supporting_nodes = self.get_sn_random(R, row_names)

        # 度
        # supporting_nodes = self.get_sn_degree(R, row_names)

        # 邻居的平均度
        # supporting_nodes = self.get_sn_ave_neighbor_degree(R, row_names)

        # NN
        # supporting_nodes = self.get_sn_NN(R,row_names)



        return supporting_nodes

    def get_sn_random(self, R, row_names):
        candidates = set(range(R.shape[0]))
        supporting_nodes = set()
        for s1 in range(R.shape[0]):
            if R[s1].max() == 2 and s1 in candidates:
                if random.random() >= 0.5:
                    supporting_nodes.add(row_names[s1])
                else:
                    supporting_nodes.add(row_names[R[s1].argmax()])
                candidates.remove(R[s1].argmax())
                candidates.remove(s1)
            else:
                continue
        return supporting_nodes

    def get_sn_degree(self, R, row_names):
        candidates = set(range(R.shape[0]))
        supporting_nodes = set()
        for s1 in range(R.shape[0]):
            if R[s1].max() == 2 and s1 in candidates:
                s2 = R[s1].argmax()

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

    def get_sn_ave_neighbor_degree(self, R, row_names):
        candidates = set(range(R.shape[0]))
        supporting_nodes = set()
        for s1 in range(R.shape[0]):
            if R[s1].max() == 2 and s1 in candidates:
                # s1和s2 都是supporting node
                s2 = R[s1].argmax()

                n_1 = 0
                n_2 = 0
                di_1 = 0
                di_2 = 0
                for i in range(np.size(R[s1])):
                    if R[s1, i] > 0:
                        n_1 += 1
                        di_1 += R[i].sum()
                for i in range(np.size(R[s2])):
                    if R[s2, i] > 0:
                        n_2 += 1
                        di_2 += R[i].sum()

                ave_neighbor_degree_1 = di_1 / n_1
                ave_neighbor_degree_2 = di_2 / n_2
                if ave_neighbor_degree_1 >= ave_neighbor_degree_2:
                    supporting_nodes.add(row_names[s1])
                else:
                    supporting_nodes.add(row_names[s2])
                candidates.remove(s2)
                candidates.remove(s1)
            else:
                continue
        return supporting_nodes

    def get_sn_NN(self, R, row_names):
        candidates = set(range(R.shape[0]))
        supporting_nodes = set()
        for s1 in range(R.shape[0]):
            if R[s1].max() == 2 and s1 in candidates:
                # s1和s2 都是supporting node
                s2 = R[s1].argmax()

                n_1 = 0
                n_2 = 0
                di_1 = 0
                di_2 = 0
                for i in range(np.size(R[s1])):
                    if R[s1, i] > 0:
                        n_1 += 1
                        di_1 += R[i].sum()
                for i in range(np.size(R[s2])):
                    if R[s2, i] > 0:
                        n_2 += 1
                        di_2 += R[i].sum()

                ave_neighbor_degree_1 = di_1 / n_1
                ave_neighbor_degree_2 = di_2 / n_2
                if ave_neighbor_degree_1 >= ave_neighbor_degree_2:
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
            clustering_tree[p].add(c)
    # print(clustering_tree)
    return clustering_tree, roots


def get_height_th_tree(tree, root):
    n = 1
    ps = set([root])
    # print('root:',root)
    while len(ps) > 0:
        ps_new = set()
        for p in ps:
            if p in tree.keys() and len(tree[p]) > 0:
                ps_new = ps_new.union(tree[p])
                # print(p, '->', tree[p])
                n += len(tree[p])
                # print('n:',n)
                # print('ps_new:',ps_new)
        # print('ps:',ps)
        ps = ps_new
    # print('n=',n)
    # print('h=',math.ceil(math.log2(n)/math.log2(2)))
    return math.ceil(math.log2(n) / math.log2(1.5))
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
            d[i, j] += 0.00001 * random.random() * random.random()
    # get the name of rows
    return d, data._stat_axis.values.tolist()


def get_adjacent_matrix(data):
    d = data.values

    """
    普通的向量数据
    """
    neighbors = NearestNeighbors(n_neighbors=2)
    neighbors.fit(d)
    A = neighbors.kneighbors_graph(d) - np.eye(len(d))
    R = A + A.T

    """
    混合数据
    """
    # dts = data.dtypes
    # max_values = np.ones((dts.size))
    # for f in range(dts.size):
    #     if str(dts[f]) != 'object':
    #         max_values[f] = max(d[:,f])
    # distance = np.zeros((data.size, data.size))
    # for i in range(data.size):
    #     for j in range(i-1):
    #         for f in range(dts.size):
    #             if dts[f] is object:
    #                 if d[i,f] == d[j,f]:
    #                     distance[i,j] = distance[i,j] + 1.0
    #
    #             else:
    #                 distance += math.pow((d[i,f]-d[j,f])/max_values[f],2)
    #         distance[i, j] = math.pow(distance[i,j],2) / len(dts)
    #         distance[j, i] = distance[i, j]
    # A = distance
    # for i in range(len(A)):
    #     m = max(A[i])
    #     for j in random(len(A)):
    #         A[i,j] = np.floor((1-distance[i,j])/(1-m))
    # R = A + A.T
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

    """
    Read data
    "breast-w", "ecoli", "glass", "ionosphere", "iris", "kdd_synthetic_control", "mfeat-fourier",
    "mfeat-karhunen","mfeat-zernike",
    "optdigits", "segment", "sonar", "vehicle", "waveform-5000", "letter", "kdd_synthetic_control"
    "adult"
    """
    # # file_name = '/Users/wenboxie/Data/uci-20070111/exp/mfeat-fourier.txt'
    # file_name = '/Users/wenboxie/Data/FRS/Datasets/HTRU2/HTRU_2.csv'
    #
    # data = pd.read_csv(file_name, header=None).iloc[:, 0:-1]
    # # data = (data - data.min()+0.001) / (data.max() - data.min()+0.001)
    # label = pd.read_csv(file_name, header=None).iloc[:, -1]
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


    # for i in range(1,10):
    #     prs = PRS(data)
    #     prs.get_clusters(i)
    #     r = prs.get_results()
    #     # print('k =', len(set(r.values())))
    #     RS_labels_ = np.array(sorted(r.items(), key=lambda item: item[0]))[:, 1]
    #     if len(set(r.values())) == 1: break
    #     print(len(set(r.values())),metrics.cluster.adjusted_rand_score(label, RS_labels_))

    # f = open('/Users/wenboxie/Data/PRS_201904/1_labels.csv', 'w')
    # f.write('id,label\n')
    # for i in range(len(label)):
    #     f.write(str(i)+','+ str(label[i]) + '\n')
    # f.close()
    #
    #
    # for k in range(2,21):
    # kmeans_model = cluster_methods.KMeans(n_clusters=k, random_state=1, init = 'random').fit(data)
    # # print('ri_kmeans =')
    # print(metrics.cluster.adjusted_rand_score(label, kmeans_model.labels_))

    # agg_model = cluster_methods.AgglomerativeClustering(n_clusters=k).fit(data)
    # print('ri_GA =', metrics.cluster.adjusted_rand_score(label, agg_model.labels_))
    # print(metrics.cluster.adjusted_rand_score(label, agg_model.labels_))
    #

    #
    # birch_model = cluster_methods.Birch(n_clusters=k).fit(data)
    #
    # print(metrics.cluster.adjusted_rand_score(label, birch_model.labels_))
    # ap_model = cluster_methods.AffinityPropagation().fit(data)
    # print('ri_ap =', metrics.cluster.adjusted_rand_score(label, ap_model.labels_))




    # print(end - start)

    # 按行重排列
    # print(data.iloc[:10,:].take(np.random.permutation(10)))
    #
    # file = open(file_name, 'r')
    # for l in file.readlines():
    #     data.append(l.strip('\n').split(',')[0:4])

    # iterating(data, 2)


    ############## 测试样例 ########
    # file_name = '/Users/wenboxie/Data/random.txt'
    # data = pd.read_csv(file_name, header=None).iloc[:, :]
    #
    # for i in range(1, 2):
    #     prs = PRS(data)
    #     prs.get_clusters(i)
    #     r = prs.get_results()
    #
    #     RS_labels_ = np.array(sorted(r.items(), key=lambda item: item[0]))[:, 1]
    #
    #     for l in RS_labels_:
    #         print(l)


    f = open('/Users/wenboxie/Data/FRS/Results/Random_HTRU_2', 'a')

    for t in range(100):
        file_name = '/Users/wenboxie/Data/FRS/Datasets/HTRU2/HTRU_2.csv'

        data = pd.read_csv(file_name, header=None).iloc[:, 0:-1]
        data = (data - data.min() + 0.001) / (data.max() - data.min() + 0.001)
        label = pd.read_csv(file_name, header=None).iloc[:, -1]
        for i in range(1, 10):
            prs = PRS(data)
            prs.get_clusters(i)
            r = prs.get_results()
            # print('k =', len(set(r.values())))
            RS_labels_ = np.array(sorted(r.items(), key=lambda item: item[0]))[:, 1]
            if len(set(r.values())) == 1: break
            f.write(
                str(len(set(r.values()))) + '\t' + str(metrics.cluster.adjusted_rand_score(label, RS_labels_)) + '\n')

    f.close()

    print('x')
