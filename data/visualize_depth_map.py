import json
import os
import matplotlib.pyplot as plt
import numpy as np


def visualize(f):
    with open(f, 'r') as fp:
        yinp = json.load(fp)
    for i in range(len(yinp['data'])):
        fig = plt.figure(1)
        plt.title(yinp['label'][i])
        map = yinp['data'][i]
        print(np.array(map).shape)
        ii = plt.imshow(np.array(map)[:, :, 0], interpolation='nearest')
        plt.show()

dir = './depth_maps'

for f in os.listdir(dir):
    if 'json' in f:
        visualize(os.path.join(dir, f))