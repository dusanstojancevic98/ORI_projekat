import numpy as np
import matplotlib.pyplot as plt

DIM = 5
STEPS = 10
COLS = 17


def euclid_distance(x, y):
    return np.linalg.norm(x - y)


def manhatn(x, y):
    dist = 0
    for i in range(len(x)):
        dist += abs(x[i] - y[i])
    return dist


def load_data(skip=0):
    empty = 0
    loaded = 0
    with open("credit_card_data.csv") as f:
        lines = f.readlines()
        new_lines = []
        for i, line in enumerate(lines):
            if i < skip:
                continue
            flag = True
            split = line.split(",")
            for s in split:
                if s == "":
                    flag = False
                    empty += 1
                    break
            if flag:
                new_lines.append(split)
                loaded += 1

        print("EMPTY: {}".format(empty))
        print("LOADED: {}".format(loaded))
        data = np.empty((len(new_lines), COLS))
        for i, line in enumerate(new_lines):
            for j in range(0, COLS):
                data[i][j] = float(line[j + 1])
        return data


def standardize_data(data):
    print(data[0])
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


def train(alpha=0.6):
    x = load_data(skip=1)
    x = standardize_data(x)
    weights = np.random.random((DIM * DIM, COLS))
    for step in range(STEPS):
        print("========= STEP: {} alpha: {} =========".format(step, alpha))
        for i, row in enumerate(x):
            min_index = None
            minim = np.inf
            for j in range(DIM * DIM):
                dist = euclid_distance(weights[j], row)
                if dist < minim:
                    # print("###### MIN CHANGED: {} -> {} ######".format(min_index, j))
                    min_index = j
                    minim = dist
            if i % 100 == 0:
                print("------ row: {}, min:{} ------".format(i, min_index))
            sub = x[i] - weights[min_index]
            new_weight = weights[min_index] + alpha * sub
            # print("SUB: {}, NW:{}".format(sub, new_weight))
            weights[min_index] = new_weight

        alpha = alpha * (1 - step / STEPS)
    print(weights)

    u_matrix = np.zeros(shape=(DIM, DIM), dtype=np.float64)
    for i in range(DIM):
        for j in range(DIM):
            v = weights[i * DIM + j]
            sum_dists = 0.0
            ct = 0
            if i - 1 >= 0:  # above
                sum_dists += euclid_distance(v, weights[(i - 1) * DIM + j])
                ct += 1
            if i + 1 <= DIM - 1:  # below
                sum_dists += euclid_distance(v, weights[(i + 1) * DIM + j])
                ct += 1
            if j - 1 >= 0:  # left
                sum_dists += euclid_distance(v, weights[i * DIM + j - 1])
                ct += 1
            if j + 1 <= DIM - 1:  # right
                sum_dists += euclid_distance(v, weights[i * DIM + j + 1])
                ct += 1
            u_matrix[i][j] = sum_dists / ct
    plt.imshow(u_matrix, cmap='gray')  # black = close = clusters
    plt.show()


if __name__ == '__main__':
    train()
