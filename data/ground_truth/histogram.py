import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import plotly.plotly as py

hist = np.zeros((64,64, 3))

for object in os.listdir('./'):
    for subobject in os.listdir(object):
        for f in os.listdir(os.path.join(object, subobject)):
            try:
                if 'corr' in f:
                    print(f)
                    im = cv2.imread(f)
                    hist += cv2.resize(im, (64, 64))
            except:
                pass
data = hist.flatten()

plt.hist(data)
plt.title("Depth Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")

plt.savefig('../depth.png')