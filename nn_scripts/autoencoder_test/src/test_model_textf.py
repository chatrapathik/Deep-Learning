import sys
import json
import random

import numpy as np
from keras.models import load_model
from keras import backend as K

from scipy.spatial import distance
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

def _euclidean_dis_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=0))


dims = int(sys.argv[3])

model = load_model(sys.argv[1], custom_objects=dict(_euclidean_dis_loss=_euclidean_dis_loss))

test_d = open(sys.argv[2])

test_samples = []
count = 1
for line in test_d:
    samples = json.loads(line)
    for sample in samples:
        test_samples.append(sample)

        count += 1

        if count == 6:
            break

test_d.close()

ind_samples_in = []
ind_samples_out = []
eucd = []
for sample in test_samples:
    #print("Input samples are:", sample)
    ind_samples_in.extend(sample)

    sample = np.array(sample).reshape(1, dims)
    #print("Predicted results are:", model.predict(sample)[0])

    eucd.append(distance.euclidean(sample, model.predict(sample)[0]))
    print(len(eucd))
    ind_samples_out.extend(model.predict(sample)[0])

test_d.close()

# Plot in vs out
plt.scatter(range(len(ind_samples_in)), ind_samples_in, c='r')
plt.scatter(range(len(ind_samples_out)), ind_samples_out, c='g')
plt.savefig(sys.argv[4])

# Calculation of MAE
#print("Individual MAEs:", maes)
print("Average MAE:", np.mean(eucd))

