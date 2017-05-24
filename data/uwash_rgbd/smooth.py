import os
import time
from scipy import misc
from scipy import stats
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

def apply_style(subplot):
    subplot.tick_params(which='both', bottom='off', top='off', left='off', right='off',
                    labelbottom='off', labelleft='off')
    subplot.spines['top'].set_visible(False)
    subplot.spines['bottom'].set_visible(False)
    subplot.spines['left'].set_visible(False)
    subplot.spines['right'].set_visible(False)
    
def fix_image(args):
    file, save = args
    im = misc.imread(file)
    filter_size = 3
    pad = int((filter_size - 1) / 2)
    im = np.pad(im, pad, 'edge')
    correct = np.copy(im)
    for i in range(pad, im.shape[0] - pad):
        for j in range(pad, im.shape[1] - pad):
            if im[i, j] == 0:
                mask = im[i - pad : i + pad + 1, j - pad : j + pad + 1]

                extra_pad = pad
                while np.sum(mask) == 0:
                    extra_pad += 2
                    start_i = i - extra_pad
                    stop_i = i + extra_pad + 1
                    start_j = j - extra_pad
                    stop_j = j + extra_pad + 1

                    start_i = 0 if start_i < 0 else start_i
                    start_j = 0 if start_j < 0 else start_j

                    H, W = im.shape
                    stop_i = H - 1 if stop_i >= H else stop_i
                    stop_j = W - 1 if stop_j >= W else stop_j

                    mask = im[start_i : stop_i, start_j : stop_j]

                mode, _ = stats.mode(mask[np.nonzero(mask)])
                correct[i, j] = mode

    correct = correct[pad:-pad, pad:-pad]
    im_med = ndimage.median_filter(correct, 9)
    
    if save:
        img = misc.toimage(im_med, high=np.max(im_med), low=np.min(im_med), mode='I')
        filename = str(file).split('.')[0] + '_corr.png'
        print(filename)
        img.save(filename)

    # Invert Images for Display
    inv = 1.0 / np.copy(im_med)
    # im_med = 1.0 / im_med

    return im, correct, inv, im_med

def depth_preprocess(data_dir, num_threads, save=False):
    tasks = []
    for category in os.listdir(data_dir):
        category_dir = os.path.join(data_dir, category)

        if os.path.isdir(category_dir):
            for instance in os.listdir(category_dir):
                instance_dir = os.path.join(category_dir, instance)
                if os.path.isdir(instance_dir):
                    if not instance == '.DS_Store':
                        for obj_file in os.listdir(instance_dir):
                            if 'depthcrop.png' in obj_file:
                                depth_dir = os.path.join(instance_dir, obj_file)
                                tasks.append((depth_dir, save))

    pool = multiprocessing.Pool(num_threads)
    results = []
    r = pool.map_async(fix_image, tasks, callback=None)
    r.wait()

    return

def test_preprocessing(save=False):
    images = []
    images.append("rgbd-dataset/cereal_box/cereal_box_1/cereal_box_1_2_144_depthcrop.png")
    # images.append("rgbd-dataset/peach/peach_2/peach_2_1_1_depthcrop.png")
    # images.append("rgbd-dataset/scissors/scissors_3/scissors_3_1_1_depthcrop.png")
    # images.append("rgbd-dataset/soda_can/soda_can_4/soda_can_4_1_1_depthcrop.png")
    # images.append("rgbd-dataset/water_bottle/water_bottle_4/water_bottle_4_1_1_depthcrop.png")
    images.append("rgbd-dataset/keyboard/keyboard_2/keyboard_2_1_138_depthcrop.png")

    num_images = len(images)
    plt.figure()
    for i in range(num_images):
        im, correct, inv, im_med = fix_image((images[i], save))

        threshold = 900
        im[im > threshold] = threshold
        correct[correct > threshold] = threshold
        im_med[im_med > threshold] = threshold

        original = plt.subplot(num_images, 3, 3 * i + 1)
        apply_style(original)
        original.set_xlabel("(a) Original")
        plt.imshow(im, cmap='viridis_r')

        corrected = plt.subplot(num_images, 3, 3 * i + 2)
        apply_style(corrected)
        corrected.set_xlabel("(b) Corrected")
        plt.imshow(correct, cmap='viridis_r')

        smoothed = plt.subplot(num_images, 3, 3 * i + 3)
        apply_style(smoothed)
        smoothed.set_xlabel("(c) Smoothed")
        plt.imshow(im_med, cmap='viridis_r')

    plt.tight_layout()
    plt.show()

def main():
    # test_preprocessing(save=False)
    start = time.time()
    depth_preprocess("rgbd-dataset", 48, save=True)
    end = time.time()
    print("Completed in %f seconds!" % (end - start))

if __name__ == '__main__':
    main()
    exit(0)
