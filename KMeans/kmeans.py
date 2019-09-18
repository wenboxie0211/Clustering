import timeit

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


class KMeansBase:
    def __init__(self, data, k):
        self.data = data
        self.k = k

    def cluster(self):
        return self._lloyds_iterations()

    def _initial_centroids(self):
        # get the initial set of centroids
        # get k random numbers between 0 and the number of rows in the data set
        centroid_indexes = np.random.choice(range(self.data.shape[0]), self.k, replace=False)
        # get the corresponding data points
        return self.data[centroid_indexes, :]

    def _lloyds_iterations(self):
        # warnings.simplefilter("error")
        centroids = self._initial_centroids()
        # print('Initial Centroids:', centroids)

        stabilized = False

        j_values = []
        iterations = 0
        while (not stabilized) and (iterations < 1000):
            print('iteration counter: ', iterations)
            try:
                # find the Euclidean distance between a center and a data point
                # centroids array shape = k x m
                # data array shape = n x m
                # In order to broadcast it, we have to introduce a third dimension into data
                # data array becomes n x 1 x m
                # now as a result of broadcasting, both array sizes will be n x k x m
                data_ex = self.data[:, np.newaxis, :]
                euclidean_dist = (data_ex - centroids) ** 2
                # now take the summation of all distances along the 3rd axis(length of the dimension is m).
                # This will be the total distance from each centroid for each data point.
                # resulting vector will be of size n x k
                distance_arr = np.sum(euclidean_dist, axis=2)

                # now we need to find out to which cluster each data point belongs.
                # Use a matrix of n x k where [i,j] = 1 if the ith data point belongs
                # to cluster j.
                min_location = np.zeros(distance_arr.shape)
                min_location[range(distance_arr.shape[0]), np.argmin(distance_arr, axis=1)] = 1

                # calculate J
                j_val = np.sum(distance_arr[min_location == True])
                j_values.append(j_val)

                # calculates the new centroids
                new_centroids = np.empty(centroids.shape)
                for col in range(0, self.k):
                    if self.data[min_location[:, col] == True, :].shape[0] == 0:
                        new_centroids[col] = centroids[col]
                    else:
                        new_centroids[col] = np.mean(self.data[min_location[:, col] == True, :], axis=0)

                # compare centroids to see if they are equal or not
                if self._compare_centroids(centroids, new_centroids):
                    # it has resulted in the same centroids.
                    stabilized = True
                else:
                    centroids = new_centroids
            except:
                print('exception!')
                continue
            else:
                iterations += 1

        print('Required ', iterations, ' iterations to stabilize.')
        return iterations, j_values, centroids, min_location

    def _compare_centroids(self, old_centroids, new_centroids, precision=-1):
        if precision == -1:
            return np.array_equal(old_centroids, new_centroids)
        else:
            diff = np.sum((new_centroids - old_centroids) ** 2, axis=1)
            if np.max(diff) <= precision:
                return True
            else:
                return False

    def initCost(self):
        t = timeit.Timer(lambda: self._initial_centroids())
        return t.timeit(number=10)


class KMeansPP(KMeansBase):
    def __init__(self, data, k):
        KMeansBase.__init__(self, data, k)

    def _initial_centroids(self):
        # pick the initial centroid randomly
        centroids = self.data[np.random.choice(range(self.data.shape[0]), 1), :]
        data_ex = self.data[:, np.newaxis, :]

        # run k - 1 passes through the data set to select the initial centroids
        while centroids.shape[0] < self.k:
            # print (centroids)
            euclidean_dist = (data_ex - centroids) ** 2
            distance_arr = np.sum(euclidean_dist, axis=2)
            min_location = np.zeros(distance_arr.shape)
            min_location[range(distance_arr.shape[0]), np.argmin(distance_arr, axis=1)] = 1
            # calculate J
            j_val = np.sum(distance_arr[min_location == True])
            # calculate the probability distribution
            prob_dist = np.min(distance_arr, axis=1) / j_val
            # select the next centroid using the probability distribution calculated before
            centroids = np.vstack(
                [centroids, self.data[np.random.choice(range(self.data.shape[0]), 1, p=prob_dist), :]])
        return centroids


if __name__ == '__main__':
    k = 3
    data = np.random.randn(100000, 2)
    # data = np.array([[1.1,2],[1,2],[0.9,1.9],[1,2.1],[4,4],[4,4.1],[4.2,4.3],[4.3,4],[9,9],[8.9,9],[8.7,9.2],[9.1,9]])
    kmeans = KMeansPP(data, k)
    _, _, centroids, min_location = kmeans.cluster()
    print(min_location)
    # plotting code
    plt.figure()
    plt.subplot(1, 3, 1)
    colors = iter(cm.rainbow(np.linspace(0, 1, k + 1)))
    for col in range(0, k):
        plt.scatter(data[min_location[:, col] == True, :][:, 0], data[min_location[:, col] == True, :][:, 1],
                    color=next(colors))

    centroid_leg = plt.scatter(centroids[:, 0], centroids[:, 1], color=next(colors), marker='x')
    plt.legend([centroid_leg], ['Centroids'], scatterpoints=1, loc='best')

    kmeans = KMeansBase(data, k)
    _, _, centroids, min_location = kmeans.cluster()
    plt.subplot(1, 3, 2)
    colors = iter(cm.rainbow(np.linspace(0, 1, k + 1)))
    for col in range(0, k):
        plt.scatter(data[min_location[:, col] == True, :][:, 0], data[min_location[:, col] == True, :][:, 1],
                    color=next(colors))

    centroid_leg = plt.scatter(centroids[:, 0], centroids[:, 1], color=next(colors), marker='x')
    plt.legend([centroid_leg], ['Centroids'], scatterpoints=1, loc='best')

    # kmeans = ScalableKMeansPP(data, k, 2, 2)
    # _, _, centroids, min_location = kmeans.cluster()
    # plt.subplot(1,3,3)
    # colors = iter(cm.rainbow(np.linspace(0, 1, k + 1)))
    # for col in range (0,k):
    #         plt.scatter(data[min_location[:,col] == True, :][:,0], data[min_location[:,col] == True, :][:,1], color=next(colors))
    #
    # centroid_leg = plt.scatter(centroids[:,0], centroids[:,1], color=next(colors), marker='x')
    # plt.legend([centroid_leg], ['Centroids'], scatterpoints=1, loc='best')
    #
    plt.show()
