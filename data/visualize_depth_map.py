import json
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def visualize(f):
    with open(f, 'r') as fp:
        yinp = json.load(fp)
    for i in range(len(yinp['data'])):
        fig = plt.figure(1)
        plt.title(yinp['label'][i])
        map = yinp['data'][i]
        ii = plt.imshow(np.array(map)[:, :, 0], interpolation='nearest')
        plt.colorbar()
        fig = plt.figure(2)
        plt.title(yinp['label'][i])
        map = yinp['X'][i]
        ii = plt.imshow(np.array(map).astype('uint8'))
        plt.colorbar()
        plt.show()

dir = './depth_maps/depth_maps'

for f in os.listdir(dir):
    if 'json' in f:
        visualize(os.path.join(dir, f))