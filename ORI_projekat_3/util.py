import numpy as np


def euclid_distance(x, y):
    return np.sqrt(np.sum(np.square(x - y)))


def manhatn(x, y, x1, y1):
    dist = 0
    dist += abs(x - x1)
    dist += abs(y - y1)
    return dist


def euclid(x, y, x1, y1):
    return (x - x1) ** 2 + (y - y1) ** 2


def load_data(skip=0, cols=range(COLS)):
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
            for j in range(len(cols)):
                data[i][j] = float(line[cols[j]])
        return data



