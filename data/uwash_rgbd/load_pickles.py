import numpy as np
import os
import pickle
import random


def split_data(X, y, depth, size=None):
    # Shuffle Data
    if size:
        permutation = random.sample(range(y.shape[0]), size)
        assert len(set(permutation)) == size
    else:
        permutation = random.sample(range(y.shape[0]), y.shape[0])
        assert len(set(permutation)) == y.shape[0]

    X_shuffled = X[permutation]
    y_shuffled = y[permutation]

    # save memory
    X, y = None, None

    # Create Training Set
    train_size = int(0.7 * y_shuffled.shape[0])
    X_train = X_shuffled[range(train_size)]
    y_train = y_shuffled[range(train_size)]
    X_train_hflip = X_train[:, :, ::-1, :]
    X_train = np.concatenate((X_train, X_train_hflip), axis=0).astype("float")
    y_train = np.concatenate((y_train, y_train), axis=0)

    # Create Validation Set
    val_size = int(0.15 * y_shuffled.shape[0])
    X_val = X_shuffled[range(train_size, train_size + val_size)].astype("float")
    y_val = y_shuffled[range(train_size, train_size + val_size)]

    # Create Test Set
    X_test = X_shuffled[
        range(train_size + val_size, y_shuffled.shape[0])].astype("float")
    y_test = y_shuffled[range(train_size + val_size, y_shuffled.shape[0])]

    # Check Depth Requirement
    if (depth):
        X_train = X_train[:, :, :, 0:3]
        X_val = X_val[:, :, :, 0:3]
        X_test = X_test[:, :, :, 0:3]

    # Normalize the Data
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Save Data in Dictionary
    data = {}
    data['X_train'] = X_train
    data['y_train'] = y_train
    data['X_val'] = X_val
    data['y_val'] = y_val
    data['x_test'] = X_test
    data['y_test'] = y_test

    return data


def load_uwash_rgbd(depth=False, size=None):
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

    data = split_data(np.concatenate(X), np.array(Y), depth, size=size)
    return data
