import argparse
import os
import sys

import h5py
import numpy as np
from PIL import Image


def format_pixel_data(pixel_array, h, w):
    list = []
    for i in range(h):
        for j in range(w):
            list.append(tuple(pixel_array[:, j, i]))
    return list


def save_original_images_to_disk(data_file_dir):
    f = h5py.File(data_file_dir, 'r')
    images = np.array(f.get("images"))
    save_images_to_disk(images, "raw-images")


def save_images_to_disk(images, root_dir):
    W, H = np.ma.size(images, axis=2), np.ma.size(images, axis=3)
    print np.ma.size(images, axis=0)
    for i in range(np.ma.size(images, axis=0)):
        print i
        im = Image.new('RGB', (W, H))
        im.putdata(format_pixel_data(images[i, :, :, :], H, W))
        im.save(os.path.join(root_dir, "%s.png" % i))


def save_single_object_images_to_disk(data_file_dir):
    f = h5py.File(data_file_dir, 'r')
    labels = ["".join([chr(j) for j in f[i].value]) for i in
              f.get("names").value[0]]
    pixel_labels = np.array(f.get("labels"))
    images = np.array(f.get("images"))
    num_images = images.shape[0]
    root_dir = "single-object-images"
    for i in range(len(labels)):
        label = labels[i]
        label_sub_dir = os.path.join(os.path.expanduser(root_dir), label)
        if not os.path.exists(label_sub_dir):
            os.makedirs(label_sub_dir)
        images_to_save = []
        for j in range(num_images):
            image = images[j]
            pixel_label = pixel_labels[j]
            if np.sum(pixel_label == (i + 1)) > 0:
                images_to_save.append(image)
        save_images_to_disk(np.array(images_to_save), label_sub_dir)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_file_dir', type=str,
                        help='Directory of the matlab data file.',
                        default='./nyu_depth_v2_labeled.mat')
    return parser.parse_args(argv)


if __name__ == '__main__':
    dir = parse_arguments(sys.argv[1:]).data_file_dir
    # save_original_images_to_disk(dir)
    # save_single_object_images_to_disk(dir)
