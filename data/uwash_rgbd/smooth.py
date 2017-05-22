from scipy import misc
from scipy import stats
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt

def apply_style(subplot):
    subplot.tick_params(which='both', bottom='off', top='off', left='off', right='off',
                    labelbottom='off', labelleft='off')
    subplot.spines['top'].set_visible(False)
    subplot.spines['bottom'].set_visible(False)
    subplot.spines['left'].set_visible(False)
    subplot.spines['right'].set_visible(False)
    
def fix_image(im):
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
    inv = 1.0 / np.copy(correct)
    im_med = ndimage.median_filter(inv, 9)

    return im, correct, inv, im_med

def main():
    images = []
    images.append(misc.imread("rgbd_dataset/cereal_box/cereal_box_1/cereal_box_1_2_144_depthcrop.png"))
    images.append(misc.imread("rgbd_dataset/peach/peach_2/peach_2_1_1_depthcrop.png"))
    images.append(misc.imread("rgbd_dataset/scissors/scissors_3/scissors_3_1_1_depthcrop.png"))
    images.append(misc.imread("rgbd_dataset/soda_can/soda_can_4/soda_can_4_1_1_depthcrop.png"))
    # images.append(misc.imread("rgbd_dataset/water_bottle/water_bottle_4/water_bottle_4_1_1_depthcrop.png"))
    images.append(misc.imread("rgbd_dataset/keyboard/keyboard_2/keyboard_2_1_138_depthcrop.png"))

    num_images = len(images)
    plt.figure()
    for i in range(num_images):
        im, correct, inv, im_med = fix_image(images[i])

        original = plt.subplot(num_images, 4, 4 * i + 1)
        apply_style(original)
        original.set_xlabel("(a) Original")
        plt.imshow(im)

        corrected = plt.subplot(num_images, 4, 4 * i + 2)
        apply_style(corrected)
        corrected.set_xlabel("(b) Corrected")
        plt.imshow(correct)

        inversed = plt.subplot(num_images, 4, 4 * i + 3)
        apply_style(inversed)
        inversed.set_xlabel("(c) Inversed")
        plt.imshow(inv)

        smoothed = plt.subplot(num_images, 4, 4 * i + 4)
        apply_style(smoothed)
        smoothed.set_xlabel("(d) Smoothed")
        plt.imshow(im_med)

    plt.tight_layout()
    plt.show()

main()
exit(0)
