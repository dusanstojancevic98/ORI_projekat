from time import time

import numpy as np
from gap_statistic import OptimalK
from matplotlib import pyplot
from pandas import DataFrame
from scipy.spatial.distance import pdist

from util import sumlog, logsum, load_data, COLS, euclid_distance, euclid_distance_squared

from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score

import seaborn as sns


class Result:
    def __init__(self, d, groups=None, copy=False, data_frame: DataFrame = None):
        if groups is None:
            groups = []
        self.groups = groups
        self.data = d
        self.points = None
        self.data_frame = data_frame
        self.copy_data(copy)

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
        self.copy_data(True)

    def copy_data(self, copy):
        if copy:
            self.np_groups = []
            for g in self.groups:
                npg = np.empty((len(g.dots_indices), COLS))
                for i, dot in enumerate(g.dots_indices):
                    npg[i][:] = data[dot]
                self.np_groups.append(npg)


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
        # print("Initialized center no.{}".format(ni + 2))
    # print("* Initialized {} cores*".format(len(groups)))
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
    if (show):
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
        if (show):
            print("Iter: {}, diff: {}".format(iter, diff))
        iter += 1

    return Result(data, groups, copy=True)


def KMeans_to_kmeans(data, nodes):
    KM = KMeans(n_clusters=nodes).fit(data)
    c = KM.cluster_centers_
    l = KM.predict(data)
    groups = []
    for ci in c:
        groups.append(Group(ci, data))
    for i, label in enumerate(l):
        groups[label].add_dot_indice(i)
    return Result(data, groups)


def kmeans_adjusted(data, nodes):
    r = kmeans(data, nodes)
    labels = np.empty(len(data)).astype(int)
    centers = np.empty((len(r.groups), COLS))
    for index, g in enumerate(r.groups):
        for i in g.dots_indices:
            labels[i] = index
        centers[index] = g.center
    # print(centers)
    # print(labels)
    return centers, labels


def sk_kmeans_adjusted(data, nodes):
    KM = KMeans(n_clusters=nodes).fit(data)
    return KM.cluster_centers_, KM.predict(data)


def create_datasets(n, data, COLS, length=None):
    n_data = len(data)
    if length == None:
        length = n_data
    data_sets = []
    mins = np.empty(COLS)
    mins.fill(np.inf)
    maxs = np.empty(COLS)
    maxs.fill(-np.inf)
    for i in range(n_data):
        for j in range(COLS):
            val = data[i][j]
            if val < mins[j]:
                mins[j] = val
            if val > maxs[j]:
                maxs[j] = val
    widths = maxs - mins
    for i in range(n):
        ds = np.empty((length, COLS))
        for j in range(length):
            for c in range(COLS):
                r = np.random.rand()
                ds[j][c] = widths[c] * r + mins[c]
        data_sets.append(ds)
        # print("Created {}. dataset".format(i+1))
    return data_sets


def gap_stat(data, n, m, func=kmeans, l=None):
    max = - np.inf
    max_k = None
    max_k_res = None
    data_sets = create_datasets(m, data, COLS, length=l)

    gaps = []
    clusters = []
    sds = []
    for ni in range(1, n + 1):
        print("===={} cores====".format(ni))
        print("----reference----")
        Wbs = []
        for ds in data_sets:
            r = func(ds, ni)
            Wbs.append(np.log(WkPDIST(r.np_groups)))
            # Wbs.append(np.log(Wk(data, r.groups)))
        E = np.mean(Wbs)
        print("----data----")

        r = func(data, ni)

        logWk = np.log(WkPDIST(r.np_groups))
        # logWk = np.log(Wk(data,r.groups))

        gap = E - logWk
        gaps.append(gap)
        clusters.append(ni)
        if gap > max:
            max_k = ni
            max_k_res = r
            max = gap

        sd = np.sqrt(np.sum([np.square(wb - E) for wb in Wbs]) / m)
        sk = np.sqrt(1. + 1. / m) * sd
        sds.append(gap - sk)
        print("---K:{} done gap_stat:{}---".format(ni, gap))
        pyplot.plot(clusters, gaps, label="gap")
        pyplot.scatter(clusters, sds, label="sd", marker="_")
        pyplot.legend()
        pyplot.show()
    pyplot.show()
    return max_k, max_k_res


def Wk(data, groups):
    Wk = 0
    for g in groups:
        Dr = 0
        nr = len(g.dots_indices)
        for i in range(nr):
            dot = g.dots_indices[i]
            for j in range(i + 1, nr):
                dot1 = g.dots_indices[j]
                if dot != dot1:
                    Dr += euclid_distance_squared(data[dot], data[dot1])
        Wk += Dr / len(g.dots_indices)
    # w = np.sum([np.mean([euclid_distance_squared(data[dot], g.center) for dot in g.dots_indices]) for g in groups])
    return Wk


def WkPDIST(groups):
    Wk = 0
    for g in groups:
        pd = pdist(g, 'sqeuclidean')
        Wk += np.sum(pd) / len(g)
    # w = np.sum([np.mean([euclid_distance_squared(data[dot], g.center) for dot in g.dots_indices]) for g in groups])
    return Wk


def wss(data, groups):
    min_dist = []
    for i in range(len(data)):
        d = data[i]
        dists = []
        for g in groups:
            dists.append(euclid_distance(g.center, d))

        min_dist.append(min(dists))
        if i % 1000 == 0 and i > 1:
            print("1000 done")
    return sum(min_dist)


def wss_grouped(data, groups):
    min_dist = []
    for g in groups:
        for i in g.dots_indices:
            d = data[i]
            min_dist.append(euclid_distance(g.center, d))

    return sum(min_dist)


def elbow(n):
    K = [i + 1 for i in range(n)]
    W = []
    for k in K:
        r = kmeans(data, k)
        W.append(wss(data, r.groups))
        print("Elbow done on {} clusters".format(k))
    pyplot.plot(K, W)
    pyplot.show()


def silhouette(data, n, func=kmeans_adjusted):
    # dist = pdist(data)
    ss = []
    ns = []
    for ni in range(2, n + 1):
        print("===={} cores====".format(ni))

        print("----kmean----")

        _, labels = func(data, ni)
        print(labels)
        s = silhouette_score(data, labels)
        print("---K:{} done s:{}---".format(ni, s))
        ss.append(s)
        ns.append(ni)
        pyplot.plot(ns, ss)
        pyplot.show()
    pyplot.show()


def hist_df(data_frame, title=None):
    fig, axes = pyplot.subplots(len(data_frame.columns) // 3 + 1, 3, figsize=(12, 48))

    fig.suptitle(title, fontsize=16)
    i = 0
    for triaxis in axes:
        for axis in triaxis:
            if i >= len(data_frame.columns):
                break
            data_frame.hist(column=data_frame.columns[i], ax=axis)
            i = i + 1
    pyplot.savefig(title + ".png")


def boxplot_df(data_frame, title=None):
    fig, axes = pyplot.subplots(len(data_frame.columns) // 3 + 1, 3, figsize=(12, 48))
    fig.suptitle(title, fontsize=16)
    i = 0
    for triaxis in axes:
        for axis in triaxis:
            if i >= len(data_frame.columns):
                break
            data_frame.boxplot(column=data_frame.columns[i], ax=axis)
            i = i + 1
    pyplot.savefig(title + ".png")
    # for col in data_frame.columns:
    #     data_frame.boxplot(column=col)


def hist_per_group(res: Result = None, title=None):
    for i, g in enumerate(res.np_groups):
        dataf = DataFrame(g, columns=res.data_frame.columns)
        # print(dataf)
        hist_df(dataf, "{}-{}-HIST-{}".format(len(res.groups), title, i))


def boxplot_per_group(res: Result = None, title=None):
    for i, g in enumerate(res.np_groups):
        dataf = DataFrame(g, columns=res.data_frame.columns)
        # print(dataf)
        boxplot_df(dataf, "{}-{}-BOXPLOT-{}".format(len(res.groups),title, i))


def boxplot_per_column(res: Result = None, title=None):
    for i, col in enumerate(res.data_frame.columns):
        fig_title = str(len(res.groups)) + "-" + title + "-BOXPLOT-" + col
        fig, axes = pyplot.subplots(1, len(res.groups), figsize=(12, 12))
        fig.suptitle(fig_title, fontsize=16)
        for j, g in enumerate(res.np_groups):
            dataf = DataFrame(g, columns=res.data_frame.columns)
            # print(dataf)
            dataf.boxplot(column=dataf.columns[i], ax=axes[j])
        pyplot.savefig(fig_title + ".png")


def correlation(data_frame):
    f, ax = pyplot.subplots(figsize=(20, 20))
    corr = data_frame.corr()
    hm = sns.heatmap(round(corr, 2), annot=True, ax=ax, cmap="coolwarm", fmt='.2f',
                     linewidths=.05)
    f.subplots_adjust(top=0.93)
    t = f.suptitle('Heatmap', fontsize=14)


if __name__ == '__main__':
    np.random.seed(2)
    # data = load_data(skip=1, cols=range(1, 18), normalize=True, norm_range=(0, 10))
    data, df = load_data(skip=1, cols=range(1, 18))
    print(df)
    # hist_df(df)
    # boxplot_df(df)
    # correlation(df)

    # k, res = gap_stat(data, 10, 10, l=100)
    # print("\n{} centers is optimal\n".format(k))
    # res.info()
    # res.show(logsum, sumlog)

    # r = kmeans(data, 7, show=True)
    # r.info()
    # r.show(logsum, sumlog)
    # r.save("KMEANS-7.save")

    # r = kmeans(data, 3, show=True)
    # r.info()
    # r.show(logsum, sumlog)
    # r.save("KMEANS-3.save")


    # print(Wk(data, r.groups))
    # print(WkPDIST(r.np_groups))

    # r = KMeans_to_kmeans(data, 8)
    # r.info()
    # r.show(logsum, sumlog)
    # r.save("KMEANStokmeans.save")

    # elbow(20)

    # r = Result(data, data_frame=df)
    # r.load("KMEANS.save")
    # r.info()
    # boxplot_per_column(r, "kmeans")
    # boxplot_per_group(r, "kmeans")
    # hist_per_group(r, "kmeans")
    # r.show(logsum, sumlog)

    # r1 = Result(data, data_frame=df)
    # r1.load("KMEANStokmeans.save")
    # r1.info()
    # boxplot_per_group(r1, "KMEANStokmeans")
    # hist_per_group(r1, "KMEANStokmeans")


    r1 = Result(data, data_frame=df)
    r1.load("KMEANS - 3.save")
    r1.info()
    boxplot_per_group(r1, "kmeans")
    # hist_per_group(r1, "kmeans")

    # oK = OptimalK(n_jobs=10, parallel_backend='joblib', clusterer=kmeans_adjusted)
    # n_clusters = oK(data, cluster_array=np.arange(1, 10), n_refs=100)
    # print(n_clusters)
    # oK.gap_df.head()
    # oK.plot_results()

    # oK = OptimalK(n_jobs=4, parallel_backend='joblib')
    # n_clusters = oK(data, cluster_array=np.arange(1, 20), n_refs=500)
    # print(n_clusters)
    # oK.plot_results()
    #
    # gap_df = oK.gap_df
    # gap_df.head()
    # ns = gap_df['n_clusters']
    # gaps = gap_df['gap_value']
    # sdks = gap_df['sdk']
    # fig, ax1 = pyplot.subplots()

    # ax2 = ax1.twinx()

    # ax1.plot(ns, sdks, 'g-')
    # ax2.plot(ns, gaps, 'b-')

    # silhouette(data, 100, sk_kmeans_adjusted)
