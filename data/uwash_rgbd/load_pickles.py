import os
import pickle
import numpy as np
import random

def split_data(X, y, depth):
    # Shuffle Data
    permutation = np.random.permutation(y.shape[0])
    X_shuffled = X[permutation]
    y_shuffled = y[permutation]

    # Create Training Set
    train_size = int(0.7 * y.shape[0])
    X_train = X_shuffled[range(train_size)]
    y_train = y_shuffled[range(train_size)]
    X_train_hflip = X_train[:, :, ::-1, :]
    X_train = np.concatenate((X_train, X_train_hflip), axis=0).astype("float")
    y_train = np.concatenate((y_train, y_train), axis=0)

    # Create Validation Set
    val_size = int(0.15 * y.shape[0])
    X_val = X_shuffled[range(train_size, train_size + val_size)].astype("float")
    y_val = y_shuffled[range(train_size, train_size + val_size)]

    # Create Test Set
    X_test = X_shuffled[range(train_size + val_size, y.shape[0])].astype("float")
    y_test = y_shuffled[range(train_size + val_size, y.shape[0])]

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

def load_uwash_rgbd(depth=False):
    X = []
    Y = []
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), './pickles')
    pickles = os.listdir(base)
    index = 0
    dict = {}
    data = {}
    data['X_train'] = []
    data['y_train'] = []
    data['X_test_val'] = []
    data['y_test_val'] = []
    for piggle in pickles:
        cucumbers = os.listdir(base + "/" + piggle)
        training_set_indices = random.sample(range(0, len(cucumbers)), 2)
        cucumber_count = 0
        for cucumber in cucumbers:
            if "pkl" in cucumber:
                pkl = open(os.path.join(base, piggle, cucumber))
                x = pickle.load(pkl)
                y = pickle.load(pkl)
                if cucumber_count in training_set_indices:
                    data['X_train'].append(x)
                    data['y_train'] += [index for i in range(x.shape[0])]
                else:
                    data['X_test_val'].append(x)
                    data['y_test_val'] += [index for i in range(x.shape[0])]
                cucumber_count += 1
                dict[index] = y
                pkl.close()
        index += 1
    data['X_train'] = np.concatenate(data['X_train'])
    data['X_test_val'] = np.concatenate(data['X_test_val'])
    if (depth):
        data['X_train'] = data['X_train'][:, :, :, 0:3]
        data['X_test_val'] = data['X_test_val'][:, :, :, 0:3]

    mean_image = np.mean(data['X_train'], axis=0)
    data['X_train'] -= mean_image
    data['X_test_val'] -= mean_image
    # data = split_data(np.concatenate(X), np.array(Y), depth)
    return data
