import numpy as np

from ORI_projekat_3.util import euclid_distance, load_data
from matplotlib import pyplot
from pandas import DataFrame


class Cluster:
    def __init__(self, data, epsilon):
        self.pts = []
        self.data = data
        self.epsilon = epsilon

    def add_point(self, point):
        self.pts.append(point)

    def check_connection(self, new_pt):
        if len(self.pts) < 0:
            return True
        for pt in self.pts:
            if euclid_distance(self.data[pt], new_pt) <= self.epsilon:
                return True
        return False

    def __len__(self):
        return len(self.pts)


class Result:
    def __init__(self, clusters=None, noise=None):
        self.clusters = clusters
        self.noise = noise

    def show(self, function_x, function_y, noise=True):
        df = {"x": [], "y": [], "Category": []}
        i = 0
        for cluster in self.clusters:
            for pt in cluster.pts:
                df["x"].append(function_x(data[pt]))
                df["y"].append(function_y(data[pt]))
                df["Category"].append(i)
            i += 1
        if noise:
            for pt in self.noise.pts:
                df["x"].append(function_x(data[pt]))
                df["y"].append(function_y(data[pt]))
                df["Category"].append(i)
        dataframe = DataFrame(df)
        groups = dataframe.groupby("Category")

        for name, group in groups:
            pyplot.scatter(group["x"], group["y"], label=name)

        pyplot.legend()

        pyplot.show()


def dbscan(data, epsilon, minpts):
    indices = [i for i in range(len(data))]
    print(len(indices))

    noise_cluster = Cluster(data, epsilon)
    for i in indices:
        neighbours = 0
        for j in indices:
            if euclid_distance(data[i], data[j]) < epsilon:
                neighbours += 1
        if neighbours < minpts:
            noise_cluster.add_point(i)
        if len(noise_cluster.pts) % 100 == 0:
            print("Added 100 more noise points.")

    for n in noise_cluster.pts:
        indices.remove(n)

    print("Removed {} points as noise".format(len(noise_cluster.pts)))

    clusters = []

    while len(indices) > 0:
        c = Cluster(data, epsilon)
        pt = indices[0]
        del indices[0]
        c.add_point(pt)
        added_index = pt
        while added_index is not None:
            n_indices = [i for i in indices]
            added_index = None
            added = None
            for i, ind in enumerate(n_indices):
                if c.check_connection(i):
                    added = ind
                    added_index = i
                    break
            if added is not None:
                c.add_point(added)
                del indices[added_index]
            else:
                break
        clusters.append(c)
        print("Added cluster with {} points.".format(len(c)))
    return Result(clusters=clusters, noise=noise_cluster)


def log2(d):
    v = 0
    try:
        v = np.sum(np.log2(d))
    except:
        pass
    return v


def sumsquared(d):
    return np.sum(np.square(d))


def squaredsum(d):
    return np.square(np.sum(d))


def product(d):
    return np.product(d)


def sum_array(d):
    return np.sum(d)


if __name__ == '__main__':
    data = load_data(skip=1, cols=range(1, 18))

    r = dbscan(data, 500, 3)

    print(len(r.clusters))

    r.show(sumsquared, log2, noise=False)
