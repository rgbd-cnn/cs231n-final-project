import os
import pickle

import numpy as np


def load_data():
    X = []
    Y = []
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), './pickles')
    pickles = os.listdir(base)
    index = 0
    dict = {}
    for piggle in pickles:
        if "pkl" in piggle:
            print('loading', piggle)
            pkl = open(os.path.join(base, piggle))
            x = pickle.load(pkl)
            X.append(x)
            y = pickle.load(pkl)
            Y += [index for i in range(x.shape[0])]
            index += 1
            dict[index] = y
            pkl.close()
    return np.concatenate(X), np.array(Y), dict
