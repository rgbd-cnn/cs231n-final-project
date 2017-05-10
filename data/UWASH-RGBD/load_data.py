import numpy as np
import os
import argparse
from PIL import Image


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str,
                        help='Directory of the UWash dataset.',
                        default='./rgbd-dataset')
    parser.add_argument('--target_img_height', type=int,
                        help='Target image height with which to resize images',
                        default=256)
    parser.add_argument('--target_img_width', type=int,
                        help='Target image width with which to resize images',
                        default=256)
    parser.add_argument('--save_resized_images_to_disk', type=bool,
                        help='whether to save resized images to disk',
                        default=False)
    return parser.parse_args(argv)


def is_rgb_file(file):
    return "crop.png" in file and "depthcrop.png" not in file and \
           "maskcrop.png" not in file


def read_and_resize_image(file_dir, depth_dir, height, width):
    desired_dimension = (width, height)
    im = Image.open(file_dir)
    out = im.resize(desired_dimension, Image.ANTIALIAS)
    rgb = np.array(out)

    im = Image.open(depth_dir)
    out = im.resize(desired_dimension, Image.ANTIALIAS)
    depth = np.array(out)

    return np.concatenate((rgb, depth), axis=2)


def save_file(data_dir, object, folder, rgb_file, depth_file, image):
    resized_object_dir = os.path.join(data_dir, object + "_resized")
    resized_folder_dir = os.path.join(resized_object_dir, folder)
    if not os.path.exists(resized_object_dir):
        os.mkdir(resized_object_dir)
    if not os.path.exists(resized_folder_dir):
        os.mkdir(resized_folder_dir)
    rgb_dir = os.path.join(resized_folder_dir, rgb_file)
    depth_dir = os.path.join(resized_folder_dir, depth_file)

    im = Image.fromarray(np.array(image[:,:,:3]))
    im.save(rgb_dir)

    im = Image.fromarray(np.array(image[:,:,3]))
    im.save(depth_dir)
    print("Saved to disk: %s and %s" % (rgb_dir, depth_dir))


def get_np_arrays_from_dataset(data_dir, height, width, save):
    X = []
    Y = []
    for object in os.listdir(data_dir):
        object_dir = os.path.join(data_dir, object)
        if os.isdir(object_dir):
            for folder in os.listdir(object_dir):
                folder_dir = os.path.join(object_dir, folder)
                if os.isdir(folder_dir):
                    dirs = os.listdir(folder_dir)
                    for file in dirs:
                        if is_rgb_file(file):
                            depth_file = file[:-8] + "depthcrop.png"
                            if depth_file in dirs:
                                file_dir = os.path.join(folder_dir, file)
                                depth_dir = os.path.join(folder_dir, depth_file)
                                x = read_and_resize_image(file_dir, depth_dir,
                                                          height, width)
                                X.append()
                                Y.append(object)
                                print("Loaded: %s, %s" % (file_dir, depth_dir))
                                if save:
                                    save_file(data_dir, object, folder, file,
                                              depth_file, x)





    return np.array(X), Y

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    data_dir = os.path.expanduser(args.data_dir)
    height = args.target_img_height
    width = args.target_img_width
    save = args.save_resized_images_to_disk
    X, Y = get_np_arrays_from_dataset(data_dir, height, width, save)


