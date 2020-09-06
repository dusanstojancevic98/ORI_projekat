import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

from ORI_projekat_3.util import euclid_distance, load_data, euclid

DIM = 50
STEPS = 10
# COLS = 17
COLS = 17
np.random.seed(1)
DIST = 30
ALPHA_MAX = 0.6


def standardize_data(data):
    max = np.full(COLS, -np.inf)
    min = np.full(COLS, np.inf)
    for i, row in enumerate(data):
        for j, d in enumerate(row):
            if d > max[j]:
                max[j] = d
            if d < min[j]:
                min[j] = d
    # print(max)
    # print(min)
    props = np.empty(COLS)
    for i in range(COLS):
        if max[i] <= 1. and min[i] >= 0.:
            props[i] = None
        else:
            props[i] = max[i] - min[i]
    # print("PROPS",props)
    for i, row in enumerate(data):
        for j, d in enumerate(row):
            if np.isnan(props[j]):
                data[i][j] = d
            elif props[j] == 0:
                data[i][j] = 1
            else:
                data[i][j] = d / props[j]
    return data


def find_nearest(row, weights):
    min_j = None
    min_k = None
    minim = np.inf
    for j in range(DIM):
        for k in range(DIM):
            dist = euclid_distance(weights[j][k], row)
            if dist < minim:
                # print("###### MIN CHANGED: {} -> {} ######".format(min_index, j))
                min_j = j
                min_k = k
                minim = dist
    return min_j, min_k


def show_u_matrix(weights, fig, step, vec):
    u_matrix = np.zeros(shape=(DIM, DIM), dtype=np.float64)
    for i in range(DIM):
        for j in range(DIM):
            v = weights[i][j]
            sum_dists = 0.0
            ct = 0
            if i - 1 >= 0:  # above
                sum_dists += euclid_distance(v, weights[i - 1][j])
                ct += 1
            if i + 1 <= DIM - 1:  # below
                sum_dists += euclid_distance(v, weights[i + 1][j])
                ct += 1
            if j - 1 >= 0:  # left
                sum_dists += euclid_distance(v, weights[i][j - 1])
                ct += 1
            if j + 1 <= DIM - 1:  # right
                sum_dists += euclid_distance(v, weights[i][j + 1])
                ct += 1
            u_matrix[i][j] = sum_dists / ct
    # fig.add_subplot(5, 4, (step % 5)*4 + vec % 4+1)
    # plt.ion()
    plt.imshow(u_matrix, cmap='gray')  # black = close = clusters
    plt.show()


def train(weights=None):
    fig = plt.figure(figsize=(8, 8))
    # x, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1,
    #                            random_state=4)

    x = load_data(skip=1, cols=range(1, 18))
    x = standardize_data(x)
    print(x)
    if weights is None:
        weights = np.random.random((DIM, DIM, COLS))
    for step in range(STEPS):
        percent = 1 - step / STEPS
        alpha = ALPHA_MAX * percent
        curr_range = percent * DIST
        print("========= STEP: {} alpha: {} range: {}=========".format(step, alpha, range))
        for i, row in enumerate(x):
            min_j, min_k = find_nearest(row, weights)
            if i % 1000 == 0:
                print("------ row: {}, min:{}, {} ------".format(i, min_j, min_k))
                show_u_matrix(weights, fig, step, i)
            for j in range(DIM):
                for k in range(DIM):
                    if euclid(j, k, min_j, min_k) < curr_range:
                        # print("SUB: {}, NW:{}".format(sub, new_weight))
                        weights[j][k] = weights[j][k] + alpha * (row - weights[j][k])
        show_u_matrix(weights, fig, step, STEPS)
    np.save("weights", weights)
    print(weights)

    show_u_matrix(weights, fig, STEPS, DIM * DIM)

    plt.show()


if __name__ == '__main__':
    train()
    # train2()
