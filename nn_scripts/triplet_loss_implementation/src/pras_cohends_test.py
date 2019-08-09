import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, concatenate
from keras.optimizers import RMSprop
from keras import backend as K
from keras import regularizers
import keras.optimizers as optimizers
import tensorflow as tf

rnd = lambda x: np.random.uniform(low=-1.0, high=1.0, size=(x, 300)).astype(np.float32)

N = 500
same_p1 = rnd(N)
same_p2 = rnd(N)
nsame_p1 = rnd(N)
nsame_p2 = rnd(N)


a = 'softsign'

m = Sequential()
m.add(Dense(300, input_shape=(300,), activation=a, activity_regularizer=regularizers.L1L2()))
m.add(Dense(150, activation=a))
m.add(Dense(250, activation=a))
m.add(Dense(300, activation='sigmoid'))

w = m.get_weights()
for x in w:
    x.fill(0)

import pdb; pdb.set_trace()

def nmap(vecs):
    for i in range(len(vecs)):
        vecs[i] = m.predict(vecs[i:i+1])

nmap(same_p1)
nmap(same_p2)
nmap(nsame_p1)
nmap(nsame_p2)

euc = lambda p1, p2: np.sqrt(np.sum((p1-p2)**2, axis=1))
same_euc = euc(same_p1, same_p2)
nsame_euc = euc(nsame_p1, nsame_p2)

smean, sstd = np.mean(same_euc), np.std(same_euc)
nsmean, nsstd = np.mean(nsame_euc), np.std(nsame_euc)

d = abs((smean - nsmean) / ((sstd + nsstd) / 2.))

print(smean, sstd, nsmean, nsstd)
print(d)
