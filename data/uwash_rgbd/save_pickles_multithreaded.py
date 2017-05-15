import argparse
import multiprocessing
import numpy as np
import os
import pickle
import sys
from PIL import Image

import load_pickles


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
    parser.add_argument('--overwrite', type=bool,
                        help='whether to rewrite existing pickles',
                        default=False)
    return parser.parse_args(argv)


def is_rgb_file(file):
    return "crop.png" in file and "depthcrop.png" not in file and \
           "maskcrop.png" not in file


def read_and_resize_image(file_dir, depth_dir, height, width):
    xs, rgbs, depths, suffices = [], [], [], []
    desired_dimension = (width, height)

    im = Image.open(file_dir)
    out_rgb = im.resize(desired_dimension, Image.ANTIALIAS)
    im = Image.open(depth_dir)
    out_depth = im.resize(desired_dimension, Image.ANTIALIAS)

    # original_image
    rgb = np.array(out_rgb)
    depth = np.reshape(np.array(out_depth), (height, width, 1))

    xs.append(np.concatenate((rgb, depth), axis=2))
    rgbs.append(rgb)
    depths.append(depth)
    suffices.append("_original")

    # vertically flipped image
    # rgb_vertical = np.flip(rgb, 0)
    # depth_vertical = np.flip(depth, 0)
    #
    # xs.append(np.concatenate((rgb_vertical, depth_vertical), axis=2))
    # rgbs.append(rgb_vertical)
    # depths.append(depth_vertical)
    # suffices.append("_vertical_flip")
    return xs, rgbs, depths, suffices


def save_file(data_dir, object, folder, rgb_file, depth_file, rgb, depth,
              suffix=""):
    resized_obj_dir = os.path.join(data_dir, object + "_resized")
    resized_folder_dir = os.path.join(resized_obj_dir, folder)
    if not os.path.exists(resized_obj_dir):
        os.mkdir(resized_obj_dir)
    if not os.path.exists(resized_folder_dir):
        os.mkdir(resized_folder_dir)

    rgb_dir = os.path.join(resized_folder_dir,
                           rgb_file[:-4] + suffix + rgb_file[-4:])
    depth_dir = os.path.join(resized_folder_dir,
                             depth_file[:-4] + suffix + depth_file[-4:])

    im = Image.fromarray(rgb, 'RGB')
    im.save(rgb_dir)

    im = Image.fromarray(depth, 'RGB')
    im.save(depth_dir)
    print("Saved to disk: %s and %s" % (rgb_dir, depth_dir))


def save_pkl(tup):
    object_dir, object, height, width, save = tup
    for folder in os.listdir(object_dir):
        X = []
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
                        xs, rgbs, depths, suffices = read_and_resize_image(
                            file_dir, depth_dir, height, width)
                        print("Loaded: %s, %s" % (file_dir, depth_dir))
                        for i in range(len(xs)):
                            x = xs[i]
                            rgb = rgbs[i]
                            depth = depths[i]
                            suffix = suffices[i]
                            X.append(x)
                            if save:
                                save_file(data_dir, object, folder, file,
                                          depth_file, rgb, depth, suffix)
        X = np.array(X)
        print("Writing pickleeeee :)")
        print(folder)
        if not os.path.exists(os.path.dirname('pickles/' + object + '/' + folder)):
            os.makedirs(os.path.dirname('pickles/' + object + '/' + folder))

        pickle_f = 'pickles/' + object + '/' + folder + '.pkl'
        with open(pickle_f, 'wb') as f:
            pickle.dump(X, f, -1)
            pickle.dump(object, f, -1)
            print("saved:", pickle_f)


def save_original_images_to_disk_as_pkls(data_dir, height, width, save,
                                         overwrite):
    tasks = []
    for object in os.listdir(data_dir):
        if "_resized" not in object and (
                overwrite or object + '.pkl' not in os.listdir('./pickles')):
            object_dir = os.path.join(data_dir, object)
            if os.path.isdir(object_dir):
                print(object_dir)
                tasks.append((object_dir, object, height, width, save))
    pool = multiprocessing.Pool(4)
    results = []
    r = pool.map_async(save_pkl, tasks, callback=results.append)
    r.wait()
    print results
    return


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    data_dir = os.path.expanduser(args.data_dir)
    height = args.target_img_height
    width = args.target_img_width
    save = args.save_resized_images_to_disk
    overwrite = args.overwrite

    if 'pickles' not in os.listdir('./'):
        os.mkdir('./pickles')

    save_original_images_to_disk_as_pkls(data_dir, height, width, save,
                                         overwrite)
    data = load_pickles.load_uwash_rgbd()
    print data
