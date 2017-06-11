import math

import matplotlib.pyplot as plt
import numpy as np

import json


def plotNNFilter(units, k, type, image):
    filters = units.shape[3]
    plt.figure(1, figsize=(30, 30))
    n_columns = 6
    n_rows = math.ceil((filters + 1) / n_columns) + 1

    plt.subplot(n_rows, n_columns, 1)
    plt.title('Original Image %s %s' % (k, type))
    if type == 'RGB':
        plt.imshow(np.array(image).astype('uint8'))
    else:
        print(np.array(image).shape)
        plt.imshow(np.array(image)[:, :, 0])

    for i in range(1, filters+1):
        plt.subplot(n_rows, n_columns, i + 1)
        plt.title('Filter %s %s %s' % (i, k, type))
        plt.imshow(units[0, :, :, i-1], interpolation="nearest", cmap="gray")


with open('first_layer.json', 'r') as f:
    j = json.load(f)

RGB, D, dict, labels = np.array(j['rgb']), np.array(j['d']), j['dict'], j['label']
image, depth = j['image'], j['depth']

for k in range(len(RGB)):
    print(dict[str(labels[k])])
    plotNNFilter(RGB[k:k + 1], dict[str(labels[k])], 'RGB', image[k])
    plt.savefig('RGB_%s' % (dict[str(labels[k])]), bbox_inches='tight')
    plotNNFilter(D[k:k + 1], dict[str(labels[k])], 'Depth', depth[k])
    plt.savefig('D_%s' % (dict[str(labels[k])]), bbox_inches='tight')


