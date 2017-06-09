import math

import matplotlib.pyplot as plt
import numpy as np

import json


def plotNNFilter(units, k, type):
    filters = units.shape[3]
    plt.figure(1, figsize=(20, 20))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i + 1)
        plt.title('Filter %s %s %s' % (i, k, type))
        plt.imshow(units[0, :, :, i], interpolation="nearest", cmap="gray")


with open('first_layer.json', 'r') as f:
    j = json.load(f)

RGB, D = np.array(j['rgb']), np.array(j['d'])

for k in range(len(RGB)):
    plotNNFilter(RGB[k:k + 1], k, 'rgb')
    plt.show()
    plotNNFilter(D[k:k + 1], k, 'depth')
    plt.show()


