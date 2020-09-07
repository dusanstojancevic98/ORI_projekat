import numpy as np

from ORI_projekat_3.util import load_data, euclid_distance, COLS


class Group:
    def __init__(self, center, data):
        self.center = center
        self.old_center = center
        self.dots_indices = []
        self.data = data

    def add_dot_indice(self, dot):
        self.dots_indices.append(dot)

    def calculate_new(self):
        self.old_center = self.center
        n = len(self.dots_indices)
        if n != 0:
            print("{} old points".format(n))
            self.center = np.zeros(COLS)
            for i in self.dots_indices:
                self.center += self.data[i]
            self.center = sum(self.data[i] for i in self.dots_indices)
            self.center /= len(self.dots_indices)
            self.clear()

    def clear(self):
        self.dots_indices = []

    def calc_err(self):
        print("Old center: {}, New center: {}".format(self.old_center, self.center))
        return np.sum(np.abs(self.center - self.old_center))

def kmeans(data, nodes, err=0.0001):
    print("Centers: {}, Err: {}".format(nodes, err))
    diff = np.inf
    groups = []

    indices = [i for i in range(len(data))]
    for i in range(nodes):
        n = np.random.randint(0, len(indices))
        groups.append(Group(data[n], data))
        indices.remove(i)
    for i, row in enumerate(data):
        min = np.inf
        imin = None
        for j, g in enumerate(groups):
            if euclid_distance(row, g.center) < min:
                imin = j
        groups[imin].add_dot_indice(i)
    while diff > err:
        for g in groups:
            g.calculate_new()

        for i, row in enumerate(data):
            min = np.inf
            imin = None
            for j, g in enumerate(groups):
                if euclid_distance(row, g.center) < min:
                    imin = j
            groups[imin].add_dot_indice(i)

        diff = sum(g.calc_err() for g in groups)
        print("Diff: {}".format(diff))

if __name__ == '__main__':
    data = load_data(skip=1, cols=range(1, 18))

    centers = kmeans(data, 10)
