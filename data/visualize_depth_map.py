import os
import matplotlib.pyplot as plt
import numpy as np


def visualize(f):
    yinp = np.load(f)
    fig = plt.figure(1)
    ii = plt.imshow(yinp[0, :, :, 0], interpolation='nearest')
    plt.show()

dir = './depth_maps'

for f in os.listdir(dir):
    if 'npy' in f:
        visualize(f)