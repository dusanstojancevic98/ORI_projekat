from ORI_projekat_3.util import euclid_distance, load_data


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


def dbscan(data, epsilon, minpts):
    indices = [i for i in range(len(data))]
    print(len(indices))

    noise = []
    for i in indices:
        neighbours = 0
        for j in indices:
            if euclid_distance(data[i], data[j]) < epsilon:
                neighbours += 1
        if neighbours < minpts:
            noise.append(i)
        if len(noise) % 100 == 0:
            print("Added 100 more noise points.")

    for n in noise:
        indices.remove(n)

    print("Removed {} points as noise".format(len(noise)))

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
    return clusters


if __name__ == '__main__':
    data = load_data(skip=1, cols=range(1, 18))

    c = dbscan(data, 10, 5)

    print(len(c))
