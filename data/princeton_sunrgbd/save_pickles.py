import os
import sys
import time
import pickle
import multiprocessing
import numpy as np
from PIL import Image

def read_and_resize_image(file_dir, depth_dir, height, width):
    xs, rgbs, depths, suffices = [], [], [], []
    desired_dimension = (width, height)

    im = Image.open(file_dir)
    out_rgb = im.resize(desired_dimension, Image.ANTIALIAS)
    im = Image.open(depth_dir)
    out_depth = im.resize(desired_dimension, Image.NEAREST)
    
    assert np.min(out_depth) >= 0
    # original_image
    rgb = np.array(out_rgb)
    depth = np.reshape(np.array(out_depth), (height, width, 1))

    xs = np.concatenate((rgb, depth), axis=2)
    rgbs.append(rgb)
    depths.append(depth)
    suffices.append("_original")

    return xs, rgbs, depths, suffices

def main():
    label_dict = {}
    y_train = []
    X_train = []
    index = 0
    count = 0
    for root, subdirs, files in os.walk('SUNRGBD'):
        label_file = 'scene.txt'
        if label_file in files:
            count += 1
            print(count)
            label_path = os.path.join(root, label_file)
            depth_path = os.path.join(root, 'depth_bfx')
            rgb_path = os.path.join(root, 'image')
            rgb_file = os.listdir(rgb_path)[0]
            depth_file = os.listdir(depth_path)[0]
            rgb_path = os.path.join(rgb_path, rgb_file)
            depth_path = os.path.join(depth_path, depth_file)
            x, _, _, _ = read_and_resize_image(rgb_path, depth_path, 64, 64)
            X_train.append(x)

            with open(label_path) as file:
                class_label = file.readline()

                if class_label in label_dict:
                    y_train.append(label_dict[class_label])
                else:
                    y_train.append(index)
                    label_dict[class_label] = index
                    index += 1

    X = np.array(X_train)
    y = np.array(y_train)
    print(X.shape)
    print(y.shape)
    print(label_dict)

    if not os.path.exists(os.path.dirname('pickles/')):
        os.makedirs(os.path.dirname('pickles/'))

    pickle_f = 'pickles/scenes.pkl'
    with open(pickle_f, 'wb') as f:
        pickle.dump(X, f, -1)
        print("Saved RGB-D Images!")

    pickle_f = 'pickles/labels.pkl'
    with open(pickle_f, 'wb') as f:
        pickle.dump(y, f, -1)
        pickle.dump(object, f, -1)
        print("Saved Labels!") 

if __name__ == '__main__':
    main()
