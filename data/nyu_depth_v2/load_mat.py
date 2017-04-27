import argparse
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


def save_images_to_disk(data_file_dir):
    f = h5py.File(data_file_dir, 'r')
    images = np.array(f.get("images"))
    W, H = np.ma.size(images, axis=2), np.ma.size(images, axis=3)
    for i in range(np.ma.size(images, axis=0)):
        print i
        im = Image.new('RGB', (W, H))
        im.putdata(format_pixel_data(images[i, :, :, :], H, W))
        im.save('raw-images/%s.png' % i)


def load_data(data_file_dir):
    f = h5py.File(data_file_dir, 'r')
    images = np.array(f.get("images"))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_file_dir', type=str,
                        help='Directory for the matlab data file.',
                        default='./nyu_depth_v2_labeled.mat')
    return parser.parse_args(argv)


if __name__ == '__main__':
    dir = parse_arguments(sys.argv[1:]).data_file_dir
    load_data(dir)
