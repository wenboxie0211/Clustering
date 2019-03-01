import numpy as np
import pandas as pd

from PRS import Core


def iterating(data: pd.DataFrame, k):

    # 数据集分割
    sub_data = Core.data_partition(data, num_sub=k)

    clusters = {}
    for i in range(len(sub_data)):
        clusters.update(Core.aggregation(sub_data[i]))

    print(clusters)
    labels = -1 * np.ones(data.shape[0])
    for root in clusters.keys():
        for node in clusters[root]:
            labels[node] = root

    # print(labels)

    #
    # fig1 = plt.figure(figsize=[10, 8])
    # ax1 = fig1.add_subplot(2, 2, 3)
    # ax1.scatter(data[0], data[1], 'o-', label='network features')
    # ax1.plot(data1['Top-N'], data1['NOR'], 's-', label='text features')
    # ax1.legend()
    # ax1.semilogx()
    # ax1.set_xlabel('Top-N')
    # ax1.set_ylabel('Precision')

    #
    data_supporting_nodes = []
    data_supporting_nodes_index = {}
    for supporting_node, i in zip(clusters.keys(), range(len(clusters.keys()))):
        data_supporting_nodes.append(data[supporting_node])
        data_supporting_nodes_index[i] = supporting_node

    clusters_supporting_nodes = Core.aggregation(data_supporting_nodes, data_supporting_nodes_index)
    for c in clusters_supporting_nodes.keys():
        for s in clusters_supporting_nodes[c]:
            clusters_supporting_nodes[c].update(clusters[s])

    return clusters_supporting_nodes


if __name__ == '__main__':
    file_name = '/Users/wenboxie/Data/uci-20070111/exp/iris.txt'
    data = pd.read_csv(file_name, header=None).iloc[:, 0:-1]

    # 按行重排列
    # print(data.iloc[:10,:].take(np.random.permutation(10)))
    #
    # file = open(file_name, 'r')
    # for l in file.readlines():
    #     data.append(l.strip('\n').split(',')[0:4])

    iterating(data, 2)
