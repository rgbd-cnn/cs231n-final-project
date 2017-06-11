import math

import matplotlib.pyplot as plt
import numpy as np

import json


def plotNNFilter(units, k, type, image):
    def ticks():
        plt.tick_params(
            axis='x',
            which='both',
            bottom='off',
            left='off',
            top='off',
            labelleft='off',
            labelbottom='off')

        plt.tick_params(
            axis='y',
            which='both',
            bottom='off',
            left='off',
            top='off',
            labelleft='off',
            labelbottom='off')

    filters = units.shape[3]
    plt.figure(1, figsize=(30, 30))
    n_columns = 8
    n_rows = 9

    if type == 'RGB':
        plt.subplot(n_rows, n_columns, 4)
        # plt.title('Original Image %s %s' % (k, type))
        ticks()
        plt.imshow(np.array(image).astype('uint8'))
    else:
        plt.subplot(n_rows, n_columns, 5)
        # plt.title('Original Image %s %s' % (k, type))
        print(np.array(image).shape)
        ticks()
        plt.imshow(np.array(image)[:, :, 0])


    for i in range(1, filters+1):
        if type == 'RGB':
            plt.subplot(n_rows, n_columns, ((i-1) // 4) * 4 + i + 8)
        else:
            plt.subplot(n_rows, n_columns, ((i-1) // 4) * 4 + i + 12)
        # plt.title('Filter %s %s %s' % (i, k, type))
        ticks()

        plt.imshow(units[0, :, :, i-1], interpolation="nearest", cmap="gray")


with open('first_layer.json', 'r') as f:
    j = json.load(f)

RGB, D, dict, labels = np.array(j['rgb']), np.array(j['d']), j['dict'], j['label']
image, depth = j['image'], j['depth']

for k in range(len(RGB)):
    print(dict[str(labels[k])])
    plotNNFilter(RGB[k:k + 1], dict[str(labels[k])], 'RGB', image[k])
    plotNNFilter(D[k:k + 1], dict[str(labels[k])], 'Depth', depth[k])
    plt.savefig(dict[str(labels[k])], bbox_inches='tight', dpi=200)


