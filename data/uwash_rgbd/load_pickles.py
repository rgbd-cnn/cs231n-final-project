import pickle
import os
import numpy as np

def load_all_the_shit():
    X = []
    Y = []
    base = os.path.dirname(os.path.abspath(__file__))
    pickles = os.listdir(base)
    for piggle in pickles:
        if "pkl" in piggle:
            pkl = open(os.path.join(base, piggle))
            X.append(pickle.load(pkl))
            Y.append(pickle.load(pkl))
            pkl.close()
    return np.concatenate(X), np.concatenate(Y)
