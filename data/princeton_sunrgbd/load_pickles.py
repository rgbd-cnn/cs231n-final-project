import os
import pickle
import numpy as np

def package_data(X, y, depth):
    # Split Data
    size = y.shape[0]
    train_size = int(0.7 * size)
    X_train = X[:train_size].astype("float")
    y_train = y[:train_size]
    X_test_val = X[train_size:].astype("float")
    y_test_val = y[train_size:]

    # Check Depth Requirement
    if not depth:
        X_train = X_train[:, :, :, 0:3]
        X_test_val = X_test_val[:, :, :, 0:3]

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
        # Invert Depth Data
        X_train[:, :, :, 3] = 1.0 / X_train[:, :, :, 3]
        X_test_val[:, :, :, 3] = 1.0 / X_test_val[:, :, :, 3]

        # Shift Training Depth Data
        min_train = np.min(X_train[:, :, :, 3])
        X_train[:, :, :, 3] -= min_train

        # Scale Training Depth Data
        max_train = np.max(X_train[:, :, :, 3])
        X_train[:, :, :, 3] *= (255.0 / max_train)

        # Shift and Scale Validation/Training Data
        X_test_val[:, :, :, 3] -= min_train
        X_test_val[:, :, :, 3] *= (255.0 / max_train)

    # Scale and Shift Data
    mean_image = np.mean(X_train, axis=0)
    std_image = np.std(X_train, axis=0)
    X_train -= mean_image
    X_train /= std_image
    X_test_val -= mean_image
    X_test_val /= std_image

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

def load_princeton(depth=False):
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), './pickles')
    scenes = os.path.join(base, 'scenes.pkl')
    labels = os.path.join(base, 'labels.pkl')
    X_pkl = open(scenes, 'r')
    y_pkl = open(labels, 'r')
    X = pickle.load(X_pkl)
    y = pickle.load(y_pkl)
    data = package_data(X, y, depth)

    return data

if __name__ == '__main__':
    data = load_princeton(depth=True)
    print(data['X_train'].shape)
    print(data['y_train'].shape)
    print(data['X_val'].shape)
    print(data['y_val'].shape)
    print(data['X_test'].shape)
    print(data['y_test'].shape)
