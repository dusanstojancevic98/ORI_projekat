import numpy as np
from pandas import DataFrame
COLS = 17


def euclid_distance(x, y):
    return np.sqrt(np.sum(np.square(x - y)))


def euclid_distance_squared(x, y):
    return np.sum(np.square(x - y))


def manhatn(x, y, x1, y1):
    dist = 0
    dist += abs(x - x1)
    dist += abs(y - y1)
    return dist


def euclid(x, y, x1, y1):
    return (x - x1) ** 2 + (y - y1) ** 2


def load_data(skip=0, cols=range(COLS), n=None, normalize=False, norm_range=None, names_index=0, ignoreNan=False):
    empty = 0
    loaded = 0
    with open("credit_card_data.csv") as f:
        lines = f.readlines()
        new_lines = []
        names = None
        for i, line in enumerate(lines):
            if i == names_index:
                names = line.strip().split(",")
            if i < skip:
                continue
            if n is not None:
                if i == n:
                    break
            flag = True
            split = line.strip().split(",")
            if ignoreNan:
                for s in split:
                    if s == "":
                        flag = False
                        empty += 1
                        break
                if flag:
                    new_lines.append(split)
                    loaded += 1
            else:
                new_lines.append(split)
                loaded += 1
        print("EMPTY: {}".format(empty))
        print("LOADED: {}".format(loaded))
        data = np.empty((len(new_lines), COLS))
        for i, line in enumerate(new_lines):
            for j in range(len(cols)):
                d = line[cols[j]]
                if d == "":
                    d = 0
                data[i][j] = float(d)
        if normalize:
            maxes = np.empty(COLS)
            maxes.fill(- np.inf)
            mins = np.empty(COLS)
            mins.fill(np.inf)

            for i in range(COLS):
                for j in range(len(data)):
                    if (data[j][i]) > maxes[i]:
                        maxes[i] = data[j][i]
                    if (data[j][i]) < mins[i]:
                        mins[i] = data[j][i]
                for j in range(len(data)):
                    frac = (data[j][i] - mins[i]) / (maxes[i] - mins[i])
                    if norm_range and norm_range[1] > norm_range[0] >= 0:
                        data[j][i] = frac * (norm_range[1] - norm_range[0]) + norm_range[0]

        if names is not None:
            data_frame = DataFrame(data=data, columns=names[1:])
        else:
            data_frame = DataFrame(data=data)
        return data, data_frame


def sumlog(d):
    v = 0
    try:
        v = np.sum(np.log2(d))
    except:
        pass
    return v


def logsum(d):
    v = 0
    try:
        v = np.log2(np.sum(d))
    except:
        pass
    return v


def log_sum_squared(d):
    v = 0
    try:
        v = np.log2(np.sum(np.square(d)))
    except:
        pass
    return v


def sin_sum(d):
    return np.sin(np.sum(d))


def sum_sin(d):
    return np.sum(np.sin(d))


def sum_cos(d):
    return np.sum(np.cos(d))


def sumsquared(d):
    return np.sum(np.square(d))


def special(d):
    v = 0
    try:
        v = np.log2(squaredsum(d) * sin_sum(d))
    except:
        pass
    return v


def squaredsum(d):
    return np.square(np.sum(d))


def product(d):
    return np.product(d)


def sum_array(d):
    return np.sum(d)
