import sys
import json
import csv
import random

import numpy as np
from keras.models import load_model
from keras import backend as K

from scipy.spatial.distance import euclidean as e
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from diskarray import DiskArray as D

def _euclidean_dis_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=0))

csv_f = open(sys.argv[4], 'w')
csv_f = csv.writer(csv_f)
csv_f.writerow(['label', 'dist_b/w_actual_vectors', 'dist_b/w_trans_vectors'])

model = load_model(sys.argv[1], custom_objects=dict(_euclidean_dis_loss=_euclidean_dis_loss))
test_f = D(sys.argv[2], dtype=[('vec1', np.float32, 300), ('vec2', np.float32, 300), ('label', np.int, 1)])

test_d = random.choice(test_f['vec2'])

inp_samples = []
out_samples = []
inp_samples.extend(test_d)
out_samples.extend(model.predict(test_d.reshape(1, 300))[0])

# Plot in vs out
fig, ax = plt.subplots()
plt.figure(figsize=(80,20))
x_ticks = [i for i in range(1, 301)]
plt.scatter(range(len(inp_samples)), inp_samples, c='r')
plt.scatter(range(len(out_samples)), out_samples, c='g')
plt.xticks(x_ticks)
plt.savefig(sys.argv[3])

for i in range(len(test_f['vec2'])):
    vec1 = test_f['vec1'][i].reshape(1, 300)
    vec2 = test_f['vec2'][i].reshape(1, 300)

    t_vec1 = model.predict(vec1)[0]
    t_vec2 = model.predict(vec2)[0]

    dist = e(vec1, vec2)
    t_dist = e(t_vec1, t_vec2)
    csv_f.writerow([test_f['label'][i], dist, t_dist])
