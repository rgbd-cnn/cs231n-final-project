import argparse
import os
import sys

import numpy as np
from PIL import Image


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_set_dir', type=str,
                        help='Directory of the test dataset.',
                        default='./SUNRGBDv2Test')
    parser.add_argument('--training_set_dir', type=str,
                        help='Directory of the training dataset.',
                        default='./SUNRGBD')
    return parser.parse_args(argv)


def get_rgbd_np_array(dir):
    print(dir)
    image_dir = os.path.join(dir, "image")
    for i in os.listdir(image_dir):
        if ".jpg" in i:
            im = Image.open(os.path.join(image_dir, i))
            image = np.array(im)
        else:
            "something is wrong in %s: %s" % (image_dir, i)
    depth_dir = os.path.join(dir, "depth")
    for i in os.listdir(depth_dir):
        if ".png" in i:
            im = Image.open(os.path.join(depth_dir, i))
            depth = np.array(im)
        else:
            "something is wrong in %s: %s" % (depth_dir, i)
    return np.concatenate((image, np.reshape(depth, (
        np.shape(depth)[0], np.shape(depth)[1], 1))), axis=2)


def get_label(dir):
    label_dir = os.path.join(dir, "scene.txt")
    with open(label_dir, 'r') as f:
        return f.read()


# X_test is 2860 x 640 x 480 x 4. 2860 is num test examples, 640 is image
# width, 480 is image height, 4 is all channels with the last being depth.
def get_rgbd_test_set(dir):
    X_test = []
    dirs = ["depth", "image"]
    num_dirs = len(dirs)
    root_dir = os.path.expanduser(dir)
    for sub_dir in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, sub_dir)):
            for sub_dir_2 in os.listdir(os.path.join(root_dir, sub_dir)):
                full_dir = os.path.join(
                    os.path.join(root_dir, sub_dir, sub_dir_2))
                if os.path.isdir(full_dir):
                    categories = os.listdir(full_dir)
                    if sum([int(i in categories) for i in dirs]) == num_dirs:
                        X_test.append(get_rgbd_np_array(full_dir))

    return X_test


# X_train is 10,355 x 640 x 480 x 4. 10,355 is num test examples, 640 is image
# width, 480 is image height, 4 is all channels with the last being depth.
# y_train is 10,355 english scene names
def get_rgbd_training_set(dir):
    X_train, y_train, types, shapes = [], [], [], []
    dirs = ["depth", "image", "scene.txt"]
    num_dirs = len(dirs)
    root_dir = os.path.expanduser(dir)
    for sub_dir in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, sub_dir)):
            for sub_dir_2 in os.listdir(os.path.join(root_dir, sub_dir)):
                full_dir_2 = os.path.join(root_dir, sub_dir, sub_dir_2)
                if os.path.isdir(full_dir_2):
                    for sub_dir_3 in os.listdir(full_dir_2):
                        full_dir = os.path.join(
                            os.path.join(full_dir_2, sub_dir_3))
                        if os.path.isdir(full_dir):
                            categories = os.listdir(full_dir)
                            if sum([int(i in categories) for i in
                                    dirs]) == num_dirs:
                                np_array = get_rgbd_np_array(full_dir)
                                X_train.append(np.array)
                                y_train.append(get_label(full_dir))
                                shapes.append(np_array.shape)
                                types.append(full_dir_2)
                            break

    return X_train, y_train


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    # X_test = get_rgbd_test_set(args.test_set_dir)
    X_train, y_train = get_rgbd_training_set(args.training_set_dir)
