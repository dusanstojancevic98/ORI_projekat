import numpy as np
from matplotlib import pyplot
from pandas import DataFrame

from util import load_data, euclid_distance, COLS, sumlog, sumsquared, logsum


class Result:
    def __init__(self, d=None, groups=None):
        if groups is None:
            groups = []
        self.groups = groups
        self.data = d

    def show(self, function_x, function_y):
        df = {"x": [], "y": [], "Category": []}
        i = 0
        for g in self.groups:
            for pt in g.dots_indices:
                df["x"].append(function_x(self.data[pt]))
                df["y"].append(function_y(self.data[pt]))
                df["Category"].append(i)
            i += 1
        dataframe = DataFrame(df)
        groups = dataframe.groupby("Category")

        for name, group in groups:
            pyplot.scatter(group["x"], group["y"], label=name, s=100)

        pyplot.legend()

        pyplot.show()

    def info(self):
        print("====Results====")
        for i, g in enumerate(self.groups):
            print("{}. group with {} points".format(i, len(g.dots_indices)))

    def save(self, name):
        with open(name, "w") as f:
            for i, g in enumerate(self.groups):
                f.write("C-{}\n".format(i))
                for pt in g.dots_indices:
                    f.write("PT-{}\n".format(pt))
                f.write("\n")

    def load(self, name):
        with open(name, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.strip()
                if line == "":
                    continue
                split = line.split("-")
                if split[0] == "C":
                    self.groups.append(Group())
                else:
                    self.groups[len(self.groups) - 1].add_dot_indice(int(split[1]))


class Group:
    def __init__(self, center=None, data=None):
        self.center = np.copy(center)
        self.old_center = self.center
        self.dots_indices = []
        self.data = data

    def add_dot_indice(self, dot):
        self.dots_indices.append(dot)

    def calculate_new(self):
        self.old_center = np.copy(self.center)
        n = len(self.dots_indices)
        if n != 0:
            self.center = np.zeros(COLS)
            for i in self.dots_indices:
                self.center += self.data[i]
            self.center = sum(self.data[i] for i in self.dots_indices)
            self.center /= len(self.dots_indices)

    def clear(self):
        self.dots_indices = []

    def calc_err(self):
        # print("Old center: {}, New center: {}".format(self.old_center, self.center))
        return np.sum(np.abs(self.center - self.old_center))


def kmeans_init(nodes, data):
    groups = []
    indices = [i for i in range(len(data))]
    for i in range(nodes):
        index = np.random.randint(0, len(indices))
        n = indices[index]
        groups.append(Group(data[n], data))
        indices.remove(n)

    return groups


def kmeanspp_init(nodes, data):
    groups = []
    indices = [i for i in range(len(data))]
    index = np.random.randint(0, len(indices))
    n = indices[index]
    groups.append(Group(data[n], data))
    indices.remove(n)
    for ni in range(nodes - 1):
        Ds = []
        Is = []
        for ii in indices:
            D = np.inf
            for g in groups:
                val = euclid_distance(data[ii], g.center)
                if val < D:
                    D = val
            if D < np.inf:
                Ds.append(D)
                Is.append(ii)
        Dmin = np.inf
        min_index = None
        for i, D in enumerate(Ds):
            if D < Dmin:
                Dmin = D
                min_index = i
        groups.append(Group(data[Is[min_index]], data))
        indices.remove(Is[min_index])
        print("Initialized center no.{}".format(ni + 2))

    return groups


def assign_dots(groups, data):
    for i, row in enumerate(data):
        min = np.inf
        imin = None
        for j, g in enumerate(groups):
            val = euclid_distance(row, g.center)
            if val < min:
                imin = j
                min = val
        groups[imin].add_dot_indice(i)
    return groups


def gap_kmeans_init(nodes, data):
    groups = kmeanspp_init(nodes, data)
    return assign_dots(groups, data)


def kmeans(data, nodes, err=0.0001, show=False):
    print("Centers: {}, Err: {}".format(nodes, err))
    diff = np.inf
    groups = kmeanspp_init(nodes, data)

    iter = 0
    while diff > err:
        for g in groups:
            g.clear()

        groups = assign_dots(groups, data)

        if show:
            r = Result(data, groups)
            r.info()
            r.show(logsum, sumlog)
        for g in groups:
            g.calculate_new()

        diff = sum(g.calc_err() for g in groups)
        print("Iter: {}, diff: {}".format(iter, diff))
        iter += 1
    return Result(data, groups)


def gap_stat(data, n, m):
    max = - np.inf
    max_k = None
    max_k_res = None
    for nii in range(n):
        ni = nii + 1
        rand_init = []
        for mi in range(m):
            rand_init.append(gap_kmeans_init(ni, data))
        E = 0
        for ri in rand_init:
            E += np.log2(Wk(data, ri))
        E /= len(rand_init)

        r = kmeans(data, ni)

        logWk = np.log2(Wk(data, r.groups))

        gap = E - logWk

        if gap > max:
            max_k = ni
            max_k_res = r
        print("K:{} done gap_stat:{}".format(nii, gap))
    return max_k, max_k_res


def Wk(data, groups):
    sse = 0
    for g in groups:
        for dot in g.dots_indices:
            sse += np.sum(np.square(data[dot], g.center))
    return sse


if __name__ == '__main__':
    np.random.seed(2)
    data = load_data(skip=1, cols=range(1, 18), normalize=True, norm_range=(0, 10))

    # k, res = gap_stat(data, 30, 100)
    # res.info()
    # res.show(logsum, sumlog)

    # r = kmeans(data, 30, show=True)
    # r.info()
    # r.show(logsum, sumlog)
    # r.save("KMEANS.save")

    r = Result(data)
    r.load("KMEANS.save")
    r.info()
    r.show(logsum, sumlog)
