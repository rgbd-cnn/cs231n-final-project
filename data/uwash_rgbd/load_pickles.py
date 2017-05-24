import os
import pickle
import numpy as np
import random
import scipy as sp

# def split_data(X, y, depth):
#     # Shuffle Data
#     permutation = np.random.permutation(y.shape[0])
#     X_shuffled = X[permutation]
#     y_shuffled = y[permutation]

#     # Create Training Set
#     train_size = int(0.7 * y.shape[0])
#     X_train = X_shuffled[range(train_size)]
#     y_train = y_shuffled[range(train_size)]
#     X_train_hflip = X_train[:, :, ::-1, :]
#     X_train = np.concatenate((X_train, X_train_hflip), axis=0).astype("float")
#     y_train = np.concatenate((y_train, y_train), axis=0)

#     # Create Validation Set
#     val_size = int(0.15 * y.shape[0])
#     X_val = X_shuffled[range(train_size, train_size + val_size)].astype("float")
#     y_val = y_shuffled[range(train_size, train_size + val_size)]

#     # Create Test Set
#     X_test = X_shuffled[range(train_size + val_size, y.shape[0])].astype("float")
#     y_test = y_shuffled[range(train_size + val_size, y.shape[0])]

#     # Check Depth Requirement
#     if not depth:
#         X_train = X_train[:, :, :, 0:3]
#         X_val = X_val[:, :, :, 0:3]
#         X_test = X_test[:, :, :, 0:3]

#     # Normalize the Data
#     mean_image = np.mean(X_train, axis=0)
#     X_train -= mean_image
#     X_val -= mean_image
#     X_test -= mean_image

#     # Save Data in Dictionary
#     data = {}
#     data['X_train'] = X_train
#     data['y_train'] = y_train
#     data['X_val'] = X_val
#     data['y_val'] = y_val
#     data['x_test'] = X_test
#     data['y_test'] = y_test

#     return data

def package_data(X_train, y_train, X_test_val, y_test_val, depth):
    # Check Depth Requirement
    if not depth:
        X_train = X_train[:, :, :, 0:3]
        X_test_val = X_test_val[:, :, :, 0:3]

    else:
        pass
        # Normalize Depth Data
        # print("Normalizing depth data...")
        # X_train[:, :, :, 3] = 1 / X_train[:, :, :, 3]
        # X_train[:, :, :, 3] *= (255.0 / np.max(X_train[:, :, :, 3], axis=(0,1,2)))
        # X_train[:, :, :, 3] -= np.mean(X_train[:, :, :, 3], axis=0)
        # X_train[:, :, :, 3] /= np.var(X_train[:, :, :, 3])
        # X_train[:, :, :, 3] *= (255.0 / np.max(X_train[:, :, :, 3], axis=(0,1,2)))
        # X_test_val[:, :, :, 3] *= (255.0 / np.max(X_test_val[:, :, :, 3], axis=(1,2)))[:, np.newaxis, np.newaxis]
        # print(np.max(X_train[:, :, :, 3]))
        # print(np.min(X_train[:, :, :, 3]))
        # print(np.mean(X_train[:, :, :, 3]))

    train_size = y_train.shape[0]
    test_val_size = y_test_val.shape[0]

    # Shuffle Training Data
    print("Shuffling training data...")
    permutation = np.random.permutation(train_size)
    X_train_shuffled = X_train[permutation]
    y_train_shuffled = y_train[permutation]

    # Create Training Set
    print("Augmenting training data...")
    X_train_hflip = X_train_shuffled[:, :, ::-1, :]
    X_train = np.concatenate((X_train_shuffled, X_train_hflip), axis=0)
    X_train_shuffled, X_train_hflip = None, None
    y_train = np.concatenate((y_train_shuffled, y_train_shuffled), axis=0)
    y_train_shuffled = None

    # Normalize the Data
    print("Normalizing data...")
    if depth:
        min_train = np.min(X_train[:, :, :, 3])
        X_train[:, :, :, 3] -= min_train

        max_train = np.max(X_train[:, :, :, 3])
        X_train[:, :, :, 3] *= (255.0 / max_train)

        X_test_val[:, :, :, 3] -= min_train
        X_test_val[:, :, :, 3] *= (255.0 / max_train)
        max_train = None
    mean_image = np.mean(X_train, axis=0)
    std_image = np.std(X_train, axis=0)
    X_train -= mean_image
    X_train /= std_image
    X_test_val -= mean_image
    X_test_val /= std_image

    # Normalize the Depth Data
    # if depth:
    #     print("Normalizing depth data...")
    #     # Scale Values
    #     max_train = np.max(X_train[:, :, :, 3], axis=(0,1,2))
    #     X_train[:, :, :, 3] *= (255.0 / max_train)
    #     X_test_val[:, :, :, 3] *= (255.0 / max_train)

        # Subtract Mean Depth Image
        # mean_depth = np.mean(X_train[:, :, :, 3], axis=0)
        # X_train[:, :, :, 3] -= mean_depth
        # X_test_val[:, :, :, 3] -= mean_depth

    # Shuffle Validation and Test Data
    print("Shuffling validation and test data...")
    permutation = np.random.permutation(test_val_size)
    X_test_val_shuffled = X_test_val[permutation]
    y_test_val_shuffled = y_test_val[permutation]
    X_test_val, y_test_val = None, None

    # Create Validation Set
    print("Creating validation set...")
    val_size = int(0.5 * float(test_val_size))
    X_val = X_test_val_shuffled[:val_size]
    y_val = y_test_val_shuffled[:val_size]

    # Create Test Set
    print("Creating test set...")
    X_test = X_test_val_shuffled[val_size:]
    y_test = y_test_val_shuffled[val_size:]

    X_test_val_shuffled, y_test_val_shuffled = None, None

    #Save Data in Dictionary
    print("Save Data in Dictionary...")
    data = {}
    data['X_train'] = X_train
    data['y_train'] = y_train
    data['X_val'] = X_val
    data['y_val'] = y_val
    data['X_test'] = X_test
    data['y_test'] = y_test

    return data

def load_uwash_rgbd(depth=False):
    X = []
    Y = []
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), './pickles')
    pickles = os.listdir(base)
    index = 0
    dict = {}
    X_train = []
    y_train = []
    X_test_val = []
    y_test_val = []
    for piggle in pickles:
        if os.path.isdir(os.path.join(base, piggle)):
            cucumbers = os.listdir(os.path.join(base, piggle))
            training_set_indices = range(1, len(cucumbers))
            cucumber_count = 0
            for cucumber in cucumbers:
                if "pkl" in cucumber:
                    # print("loading: ", cucumber)
                    pkl = open(os.path.join(base, piggle, cucumber))
                    x = pickle.load(pkl)
                    y = pickle.load(pkl)
                    if cucumber_count in training_set_indices:
                        X_train.append(x)
                        y_train += [index for i in range(x.shape[0])]
                    else:
                        X_test_val.append(x)
                        y_test_val += [index for i in range(x.shape[0])]
                    cucumber_count += 1
                    dict[index] = y
                    pkl.close()
            index += 1
    # print([i.shape for i in X_train])
    X_train = np.concatenate(X_train).astype("float")
    X_test_val = np.concatenate(X_test_val).astype("float")
    y_train = np.array(y_train)
    y_test_val = np.array(y_test_val)

    data = package_data(X_train, y_train, X_test_val, y_test_val, depth)

    # data = split_data(np.concatenate(X), np.array(Y), depth)
    return data
