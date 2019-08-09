import sys
import json
import random

import itertools
import numpy as np
from keras.models import load_model
from keras import backend as K

from scipy.spatial import distance
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from diskarray import DiskArray as D

def _euclidean_dis_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=0))


dims = int(sys.argv[3])

model = load_model(sys.argv[1], custom_objects=dict(_euclidean_dis_loss=_euclidean_dis_loss))

test_f = D(sys.argv[2], dtype=[('vecs', np.float32, 300)])
random.seed = 2
test_d = random.choice(test_f['vecs'])

inp_samples = []
out_samples = []
inp_samples.extend(test_d)
out_samples.extend(model.predict(test_d.reshape(1, dims))[0])

# Plot in vs out
fig, ax = plt.subplots()

#plt.figure(figsize=(40,20))
colors = itertools.cycle(["r", "g", "b", "c", "m", "y", "k"])

colors = cm.Dark2(np.linspace(0, 1, 300))
for i in range(300):
    clr = random.choice(colors)
    plt.scatter(i, inp_samples[i], c=clr, s=6)
    plt.scatter(i, out_samples[i], c=clr, s=6)




plt.savefig(sys.argv[4])

