import numpy as np
import time
import copy

def k_means(X, k, eps=1e-10, max_iter=1000, print_freq=10):
    """
    This function takes in the following arguments:
        1) X, the data matrix with dimension m x n
        2) k, the number of clusters
        3) eps, the threshold of the norm of the change in clusters
        4) max_iter, the maximum number of iterations
        5) print_freq, the frequency of printing the report

    This function returns the following:
        1) clusters, a list of clusters with dimension k x 1
        2) label, the label of cluster for each data with dimension m x 1
        3) cost_list, a list of costs at each iteration

    NOTE:
        1) We use l2-norm as the distance metric
    """
    m, n = X.shape
    cost_list = []
    t_start = time.time()
    # randomly generate k clusters
    clusters = np.random.multivariate_normal((.5 + np.random.rand(n))
        * X.mean(axis=0), 10 * X.std(axis=0) * np.eye(n), size=k)
    label = np.zeros((m, 1)).astype(int)
    iter_num = 0

    while iter_num < max_iter:
        prev_clusters = copy.deepcopy(clusters)

        for i in range(m):
            data = X[i, :]
            diff = data - clusters
            curr_label = np.argsort(np.linalg.norm(diff, axis=1)).item(0)
            label[i] = curr_label

            # norm_list = [(np.linalg.norm(X[i, :] - centroid))**2 for
            #              centroid in clusters]
            # norm_array = np.array(norm_list)
            # sorted_norms = np.argsort(norm_array)
            # label[i] = sorted_norms.item(0)

        for i in range(k):
            ind = np.where(label == i)[0]
            if len(ind) > 0:
                clusters[i, :] = X[ind].mean(axis=0)
            # new_centroid = np.zeros_like(clusters[0])
            # num_pts = 0
            # for j in range(len(label)):
            #     if label[j] == i:
            #         new_centroid += X[i, :]
            #         num_pts += 1
            # if num_pts > 0:
            #     clusters[i, :] = new_centroid / num_pts



        cost = k_means_cost(X, clusters, label)
        cost_list.append(cost)

        if (iter_num + 1) % print_freq == 0:
            print('-- Iteration {} - cost {:4.4E}'.format(iter_num+
                                                          1, cost))
        if np.linalg.norm(prev_clusters - clusters) <= eps:
            print('-- Algorithm converges at iteration {} with cost '
                  '{:4.4E}'.format(iter_num + 1, cost))
            break
        iter_num += 1

    t_end = time.time()
    print('-- Time elapsed: {t:2.2f} seconds'.format(t=t_end-t_start))
    return clusters, label, cost_list

def k_means_cost(X, clusters, label):
    """
    This function takes in the following arguments:
        1) X, the data matrix with dimension m x n
        2) clusters, the matrix with dimension k x 1
        3) label, the label of the cluster for each data point with
           dimension m x 1

        This function calculates and returns the cost for the given
        data and clusters.

    NOTE:
        1) The total cost is defined by the sum of the l2-norm
           difference between each data point and the cluster center
           assigned to this data point
    """
    m, n = X.shape
    k = clusters.shape[0]

    X_cluster = clusters[label.flatten()]
    cost = (np.linalg.norm(X - X_cluster, axis=1)**2).sum()

    # cost = 0.0
    # for i in range(m):
    #     cost += (np.linalg.norm(X[i] - clusters[label[i]]))**2

    return cost
