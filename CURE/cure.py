import numpy as np
from scipy.spatial import KDTree
from scipy.spatial import distance

from util import estimate


# http://www-users.cs.umn.edu/~han/dmclass/cure.pdf

class Cluster:
    def __init__(self, shape, point=None):

        if (point is not None):
            self.points = np.matrix(point)
            self.center = point
            self.rep = np.matrix(point)

        else:
            self.points = np.empty(shape=(0, shape[1]))
            self.center = None
            self.rep = np.empty(shape=(0, shape[1]))

        self.closest = None
        self.distance_closest = float('inf')


class Cure:
    def __init__(self, data, number_of_clusters, alpha, c):

        # data is a Numpy-array with rows as points and k
        # is the number of clusters
        self.data = data
        self.k = number_of_clusters
        self.alpha = alpha
        self.c = c
        # Stores representatives for each cluster
        self.KDTree = KDTree(data, leafsize = 50)

        self.shape = data.shape

        # Initializes each point as a Cluster object
        data_as_clusters = [Cluster(self.shape, point) for point in data]

        # Initializes each Clusters closest Cluster and distance using the
        # KDTree
        for cluster in data_as_clusters:
            query = self.KDTree.query(cluster.points[0], 2)
            #print query
            cluster.distance_closest = query[0][0][1]
            cluster.closest = data_as_clusters[query[1][0][1]]

        # Stores an entry for each cluster sorted by their distances to their
        # closest cluster
        self.Heap = sorted(data_as_clusters, key=lambda x:
        x.distance_closest, reverse=False)

    def cure_clustering(self):
        while (len(self.Heap) > self.k):

            # Select an arbitrary cluster from heap
            cluster_u = self.Heap[0]
            cluster_v = cluster_u.closest

            # remove to be merged elements from the heap and resort heap
            self.Heap.remove(cluster_v)
            self.Heap.remove(cluster_u)

            # merge the clusters to form a new cluster
            cluster_w = self.merge_cluster(cluster_u, cluster_v)

            tree_data = np.empty(shape=(0, self.shape[1]))

            # removing old representatives and adding representatives for new cluster
            for cluster in self.Heap:
                for rep in cluster.rep:
                    tree_data = np.concatenate((tree_data, rep))

            for rep in cluster_w.rep:
                 tree_data = np.concatenate((tree_data, rep))

            self.KDTree = KDTree(np.matrix(tree_data),leafsize = 50)

            # select arbitrary element from the heap
            cluster_w.closest = self.Heap[0]
            cluster_w.distance_closest = self.distance_func(cluster_w.center,
                                                            cluster_w.closest.center)

            for cluster in self.Heap:

                dist = self.distance_func(cluster_w.center, cluster.center)

                if (dist < cluster_w.distance_closest):
                    cluster_w.closest = cluster
                    cluster_w.distance_closest = dist

                if ((cluster.closest is cluster_u) or (cluster.closest is cluster_v)):

                    if (cluster.distance_closest < dist):
                        # get closest element to cluster with maximum distance
                        (cluster.distance_closest, cluster.closest) = self.closest_cluster(cluster, cluster_w, dist)

                    else:
                        cluster.closest = cluster_w
                        cluster.distance_closest = dist

                elif (cluster.distance_closest > dist):
                    cluster.closest = cluster_w
                    cluster.distance_closest = dist

            self.Heap.append(cluster_w)
            self.Heap.sort(key=lambda x: x.distance_closest, reverse=False)

        # finding clusterlabels relative to input data
        list_of_labels = [-999] * len(data)

        i = 0

        for c in self.Heap:
            for p in c.points:
                j = 0
                for row in self.data:
                    tet = np.squeeze(np.asarray(p))
                    if(tet == row).all():
                        list_of_labels[j] = i
                    j+=1
            i+=1

        return list_of_labels


    def merge_cluster(self, cluster1, cluster2):
        # merge points of cluster1 and cluster2
        merged_cluster = self.union_func(cluster1, cluster2)
        # calculate new mean of new cluster
        merged_cluster.center = (len(cluster1.points) * cluster1.center + len(cluster2.points) * cluster2.center) / (
        len(cluster1.points) + len(cluster2.points))
        tmpSet = []
        merged_cluster.rep = np.empty(shape=(0, self.shape[1]))
        # generate c maximum points 
        for i in range(0, self.c):
            maxDist = 0
            maxPoint = []
            for point in merged_cluster.points:
                if i == 0:
                    minDist = self.distance_func(point, merged_cluster.center)

                else:
                    tmpDist = min([self.distance_func(point, p) for p in tmpSet])
                # point is maxpoint if this is true
                if minDist >= maxDist:
                    maxDist = minDist
                    maxPoint = point

            if not any((maxPoint == x).all() for x in tmpSet):
                tmpSet.append(maxPoint)
        # calculate new representation points for merged_cluster with maxpoints
        # for i in xrange(0, len(tmpSet)):
        for i in range(0, len(tmpSet)):
            merged_cluster.rep = np.concatenate((merged_cluster.rep ,(tmpSet[i] + (merged_cluster.center - tmpSet[i]) * self.alpha)))

        return merged_cluster

    #calculate the union of 2 clusters
    def union_func(self, cluster1, cluster2):
        union_cluster = Cluster(shape=self.shape)
        union_cluster.points=np.append(cluster1.points, cluster2.points, axis=0)
        return union_cluster

    def closest_cluster(self, cluster, merged_cluster, dist):

        distance = dist
        closest_rep = []

        # getting c+1 representatives to find closest different cluster 
        for representative in cluster.rep:
            query = self.KDTree.query(representative, self.c+1, 0, 2)

        # finding closest point
        for i in range(0, self.c+1):
            if(query[0][0][i] < float('inf')):
                temp_rep = self.KDTree.data[query[1][0][i]]

                for point in cluster.rep:
                    tet = np.squeeze(np.asarray(point))
                    if (not (tet == temp_rep).all()):
                        distance = query[0][0][i]
                        closest_rep = temp_rep

        # checking if cluster is the new cluster
        for point in merged_cluster.rep:
            tet = np.squeeze(np.asarray(point))
            if(tet == closest_rep).all():
                return (distance, merged_cluster)

        # finding cluster that owns the closest point
        for clusterz in self.Heap:
            for point in clusterz.rep:
                tet = np.squeeze(np.asarray(point))
                if (tet == closest_rep).all():
                    return (distance, clusterz)

    def distance_func(self, p1, p2):
        return distance.euclidean(p1, p2)


def __load_data(path):
    f = open(path, 'r')
    n = 0
    while f.readline():
        n += 1
    f.close()
    f = open(path, 'r')

    first_line = f.readline().strip('\t\n').split('\t')

    m = len(first_line)
    data_matrix = np.zeros((n, m))

    for j in range(m):
        data_matrix[0][j] = float(first_line[j])

    for i in range(1,n):
        line = f.readline().strip('\t\n').split('\t')

        for j in range(m):
            data_matrix[i][j] = line[j]
    f.close()
    return data_matrix

def __load_label(path):
    f = open(path, 'r')

    label_list = []
    while True:
        line = f.readline()
        if not line:
            break
        label_list.append(line.strip('\n'))

    np.array(label_list)
    f.close
    return label_list

def cure_clustering(data, number_of_clusters, alpha, c):
    cure = Cure(data, number_of_clusters, alpha, c)
    return cure.cure_clustering()[:len(data)]


if __name__ == '__main__':

    #{"breast-w", "ecoli", "glass", "ionosphere", "iris", "kdd_synthetic_control", "mfeat-fourier", "mfeat-karhunen","mfeat-zernike"};
    #{"optdigits", "segment", "sonar", "vehicle", "waveform-5000", "letter", "kdd_synthetic_control"};
    # dataset = 'ERA'
    # file_name_data = '/Users/wenboxie/Data/uci-20070111/exp/' + dataset + '(data).txt'
    # file_name_label = '/Users/wenboxie/Data/uci-20070111/exp/' + dataset + '(label).txt'

    # Smaller alpha shrinks the scattered points and favors elongated clusters
    # large alph-> scattered points get closer to mean,  cluster tend to be more compact
    # alpha = 0.3
    #
    # # number of representatives per cluster
    # c = 10
    # data = __load_data(file_name_data)
    # label = __load_label(file_name_label)
    # file_results = '/Users/wenboxie/Data/rs-exp/cure/cure-' + dataset + '-ri.txt'
    # write_results = open(file_results,'w')
    # for i in range(2,21):
    #     # number of clusters
    #     number_of_clusters = i
    #
    #     # print('load data and label.')
    #     pred = cure_clustering(data, number_of_clusters, alpha, c)
    #
    #     ri = estimate.rand_index(label, pred)
    #     print(ri)
    #     write_results.write(str(ri)+'\n')

    alpha = 0.3
    file_name_data = '/Users/wenboxie/Data/ssim_cwssim.csv'
    file_name_label = '/Users/wenboxie/Data/Olivetti(label).txt'
    data = np.zeros((400, 400))
    f = open(file_name_data, 'r')
    for lin in f.readlines():
        ns = lin.strip('\n').split(',')
        data[int(ns[0]) - 1, int(ns[1]) - 1] = data[int(ns[1]) - 1, int(ns[0]) - 1] = float(ns[2])
    c = 10
    label = __load_label(file_name_label)
    file_results = '/Users/wenboxie/Data/rs-exp/cure/cure-Olivetti-ri.txt'
    write_results = open(file_results, 'w')
    for i in range(2, 50):
        # number of clusters
        number_of_clusters = i

        # print('load data and label.')
        pred = cure_clustering(data, number_of_clusters, alpha, c)

        ri = estimate.rand_index(label, pred)
        print(ri)
        write_results.write(str(ri) + '\n')
