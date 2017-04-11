import scipy.io
import numpy as np
import h5py
f = h5py.File('nyu_depth_v2_labeled.mat','r')
data = np.array(f.get("names"))
print [f[i] for i in data[0]]