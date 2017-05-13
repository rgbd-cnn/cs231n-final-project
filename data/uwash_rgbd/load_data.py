import argparse
import json
import os
import sys
import pickle

import numpy as np
from PIL import Image


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str,
                        help='Directory of the UWash dataset.',
                        default='./rgbd-dataset')
    parser.add_argument('--target_img_height', type=int,
                        help='Target image height with which to resize images',
                        default=64)
    parser.add_argument('--target_img_width', type=int,
                        help='Target image width with which to resize images',
                        default=64)
    parser.add_argument('--save_resized_images_to_disk', type=bool,
                        help='whether to save resized images to disk',
                        default=False)
    parser.add_argument('--load_from_resized_images', type=bool,
                        help='whether to load resized images saved previously',
                        default=False)
    return parser.parse_args(argv)


def is_rgb_file(file):
    return "crop.png" in file and "depthcrop.png" not in file and \
           "maskcrop.png" not in file


def read_and_resize_image(file_dir, depth_dir, height, width):
    desired_dimension = (width, height)
    im = Image.open(file_dir)
    out_rgb = im.resize(desired_dimension, Image.ANTIALIAS)
    rgb = np.array(out_rgb)

    im = Image.open(depth_dir)
    out_depth = im.resize(desired_dimension, Image.ANTIALIAS)
    depth = np.reshape(np.array(out_depth), (height, width, 1))
    return np.concatenate((rgb, depth), axis=2), rgb, depth


def save_file(data_dir, object, folder, rgb_file, depth_file, rgb, depth):
    resized_object_dir = os.path.join(data_dir, object + "_resized")
    resized_folder_dir = os.path.join(resized_object_dir, folder)
    if not os.path.exists(resized_object_dir):
        os.mkdir(resized_object_dir)
    if not os.path.exists(resized_folder_dir):
        os.mkdir(resized_folder_dir)
    rgb_dir = os.path.join(resized_folder_dir, rgb_file)
    depth_dir = os.path.join(resized_folder_dir, depth_file)

    im = Image.fromarray(rgb, 'RGB')
    im.save(rgb_dir)

    im = Image.fromarray(depth, 'RGB')
    im.save(depth_dir)
    print("Saved to disk: %s and %s" % (rgb_dir, depth_dir))


def load_original(data_dir, height, width, save):
    for object in os.listdir(data_dir):
        if "_resized" not in object:
            object_dir = os.path.join(data_dir, object)
            if os.path.isdir(object_dir):
                X = []
                Y = []
                for folder in os.listdir(object_dir):
                    folder_dir = os.path.join(object_dir, folder)
                    if os.path.isdir(folder_dir):
                        dirs = os.listdir(folder_dir)
                        for file in dirs:
                            if is_rgb_file(file):
                                depth_file = file[:-8] + "depthcrop.png"
                                if depth_file in dirs:
                                    file_dir = os.path.join(folder_dir, file)
                                    depth_dir = os.path.join(folder_dir,
                                                             depth_file)
                                    x, rgb, depth = read_and_resize_image(
                                        file_dir, depth_dir, height, width)
                                    X.append(x)
                                    Y.append(object)
                                    print(
                                        "Loaded: %s, %s" % (
                                            file_dir, depth_dir))
                                    if save:
                                        save_file(data_dir, object, folder,
                                                  file, depth_file, rgb, depth)
                X = np.array(X)
                print("Writing pickleeeee :)")
                print(object)
                with open(object + '.pkl', 'wb') as f:
                    pickle.dump(X, f, -1)
                    pickle.dump(Y, f, -1)
    return X, Y


def load_resized(data_dir, height, width):
    X = []
    Y = []
    for object in os.listdir(data_dir):
        label = object[:-8]
        if "_resized" in object:
            object_dir = os.path.join(data_dir, object)
            if os.path.isdir(object_dir):
                X = []
                Y = []
                for folder in os.listdir(object_dir):
                    folder_dir = os.path.join(object_dir, folder)
                    if os.path.isdir(folder_dir):
                        dirs = os.listdir(folder_dir)
                        for file in dirs:
                            if is_rgb_file(file):
                                depth_file = file[:-8] + "depthcrop.png"
                                if depth_file in dirs:
                                    file_dir = os.path.join(folder_dir, file)
                                    depth_dir = os.path.join(folder_dir,
                                                             depth_file)
                                    x, rgb, depth = read_and_resize_image(
                                        file_dir, depth_dir, height, width)
                                    X.append(x)
                                    Y.append(label)
                                    print(
                                        "Loaded: %s, %s" % (
                                            file_dir, depth_dir))
                X = np.array(X)
                print("Writing pickleeeee :)")
                with open(object + '.pkl', 'wb') as f:
                    pickle.dump(X, f, -1)
                    pickle.dump(Y, f, -1)
    return X, Y


def get_np_arrays_from_dataset(data_dir, height, width, save, load_resized):
    if load_resized:
        X, Y = load_resized(data_dir, height, width)
    else:
        X, Y = load_original(data_dir, height, width, save)

    return np.array(X), Y


def save_X_and_Y_json_to_disk(x, y):
    with open('data.pkl', 'wb') as f:
        pickle.dump(x, f, -1)
        pickle.dump(y, f, -1)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    data_dir = os.path.expanduser(args.data_dir)
    height = args.target_img_height
    width = args.target_img_width
    save = args.save_resized_images_to_disk
    load_resized = args.load_from_resized_images
    X, Y = get_np_arrays_from_dataset(data_dir, height, width, save,
                                      load_resized)
