import os
import sys

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


import numpy as np
from keras import models as km
from keras import layers, optimizers
from keras import backend as kb
import keras.optimizers as optimizers

from basescript import BaseScript
from diskarray import DiskArray

class ModelBuilder(object):

    EPOCHS = 100
    ACTIVATION = 'softmax'
    HIDDEN_NEURONS = 64
    OUTPUT_ACTIVATION = 'sigmoid'
    OPTIMIZER = optimizers.Adam(lr=0.1)
    #OPTIMIZER = 'rmsprop'
    LOSS = 'binary_crossentropy'
    METRICS = ['acc']

    def __init__(self, labelled_set, model_name=None, model_path=None):
        self.labelled_set = DiskArray(labelled_set,  dtype=[('diff_vec', np.float32, 301), ('label', np.int, 1)])
        self.test_set = DiskArray(sys.argv[2], dtype=[('diff_vec', np.float32, 301), ('label', np.int, 1)])
        self.model_path = model_path
        self.model_name = os.path.join(model_path, model_name)

    def make_model(self):
        t_model = km.Sequential()
        t_model.add(layers.Dense(len(self.labelled_set['diff_vec']),
                                activation=self.ACTIVATION,
                                input_shape=(301,))
                                )

        '''
        for layer in range(self.HIDDEN_LAYERS):
            t_model.add(layers.Dense(self.HIDDEN_NEURONS, activation=self.ACTIVATION))
            self.HIDDEN_NEURONS //= 2
        '''

        t_model.add(layers.Dense(1, activation=self.OUTPUT_ACTIVATION))
        t_model.compile(
            optimizer=self.OPTIMIZER,
            loss=self.LOSS,
            metrics=self.METRICS,
            )

        return t_model

    def train_model(self, model):
        vectors = self.labelled_set['diff_vec']
        labels = self.labelled_set['label']

        model.fit(vectors, labels, epochs=self.EPOCHS, shuffle=True)

        return model

    def test_model(self, model):
        loss, accuracy = model.evaluate(self.test_set['diff_vec'], self.test_set['label'])
        print('Loss = ', loss)
        print('Acc = ', accuracy)

        psame = []
        pnsame = []

        for i, vec in enumerate(self.test_set['diff_vec']):
            p_val = model.predict(vec.reshape(1, len(vec)))[0]
            if self.test_set['label'][i] == 0:
                psame.append(p_val)
            else:
                pnsame.append(p_val)

        print('Same Points Mean= ', np.mean(psame), 'Same Points STDEV= ', np.std(psame), 'Same Poinst MIN_DIST= ', np.min(psame), 'Same points MAX_DIST= ', np.max(psame))
        print('Not Same Points Mean= ', np.mean(pnsame), 'Not same Points STDEV= ', np.std(pnsame), 'Not Same Poinst MIN_DIST= ', np.min(pnsame), 'Not Same Points MAX_DIST= ', np.max(pnsame))

        dm = (np.std(psame) + np.std(pnsame)) / 2
        nm = abs(np.mean(psame) - np.mean(pnsame))

        d = nm / dm
        print(d)

        plt.hist(psame, bins=50, alpha=0.5, label='same points')
        plt.hist(pnsame, bins=50, alpha=0.5, label='not same points')
        plt.legend(loc='upper right')
        plt.savefig(sys.argv[3])

    def save(self, model):
        model.save(self.model_name)
        kb.clear_session()

    def load(self):
        t_model = km.load_model(self.model_name)

        return t_model

if __name__ == '__main__':
        mb = ModelBuilder(labelled_set=sys.argv[1], model_name='model.h5', model_path='.')
        model = mb.make_model()
        model = mb.train_model(model)
        mb.test_model(model)
        mb.save(model)
