import csv
import random

import numpy as np

from keras.models import Sequential
from keras.layers import Dense
import keras.optimizers as optimizers
from keras import regularizers
from keras.callbacks import EarlyStopping

from basescript import BaseScript
from diskarray import DiskArray

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))

class GridScript(BaseScript):

    DIMS = 301

    OPTIMIZERS = ['rmsprop', 'adam', 'sgd']
    LOSSES = ['mse', 'mae', 'binary_crossentropy']
    ACTIVATION = ['tanh', 'softmax', 'softplus', 'sigmoid', 'elu', 'selu', 'softsign', 'relu']
    EPOCHS = [100, 500, 1000, 1500, 2000]
    HIDDEN_LAYERS = [1, 2, 3, 4]

    def __init__(self):
        super(GridScript, self).__init__()
        self.train_d = DiskArray(self.args.train_d,  dtype=self._get_dtype())
        self.test_d = DiskArray(self.args.test_d,  dtype=self._get_dtype())
        self.csv = open(self.args.outf, 'w')
        self.outf = csv.writer(self.csv)
        self.outf.writerow(['num of hidden layers', 'loss', 'activcation', 'optimizer', 'epochs', 'cohens_d', 'accuracy'])
        self.hyper_parameters = []

    def _get_dtype(self):
        return [('diff_vec', np.float32, self.DIMS), ('label', np.int, 1)]


    def create_model(self, layer, loss, opt, act):
        model = Sequential()

        if layer != 1:
            model.add(Dense(self.DIMS, input_shape=(self.DIMS,), activation=act))
            for i in range(0, layer):
                neurons = int(self.DIMS / (i + 1))
                if neurons == self.DIMS: continue
                model.add(Dense(neurons, activation=act))

            model.add(Dense(1, input_shape=(self.DIMS,), activation='sigmoid'))
        else:
            model.add(Dense(1, input_shape=(self.DIMS,), activation='sigmoid'))

        model.compile(optimizer=opt, loss=loss, metrics=['acc'])

        return model

    def train_model(self, model, epoch):
        checkpoint = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1, mode='auto')
        vectors = self.train_d['diff_vec']
        labels = self.train_d['label']

        model.fit(vectors, labels, epochs=epoch, shuffle=True, callbacks=[checkpoint], validation_split=0.05)

        return model

    def process(self, layer, loss, opt, act, epoch):
        cohens_d = 0
        accuracy = 0

        for i in range(2):
            model = self.create_model(layer, loss, opt, act)
            model = self.train_model(model, epoch)
            accuracy += self.test_model(model)
            cohens_d +=  self.get_cohens_d(model)

        cohens_d = cohens_d / 2
        accuracy = accuracy / 2
        self.outf.writerow([layer, loss, act, opt, epoch, cohens_d, accuracy])
        self.csv.flush()

    def test_model(self, model):
        loss, accuracy = model.evaluate(self.test_d['diff_vec'], self.test_d['label'])
        print('Loss = ', loss)
        print('Acc = ', accuracy)

        return accuracy

    def get_cohens_d(self, model):
        psame = []
        pnsame = []

        for i, vec in enumerate(self.test_d['diff_vec']):
            p_val = model.predict(vec.reshape(1, len(vec)))[0]
            if self.test_d['label'][i] == 0:
                psame.append(p_val)
            else:
                pnsame.append(p_val)

        print('Same Points Mean= ', np.mean(psame), 'Same Points STDEV= ', np.std(psame), 'Same Poinst MIN_DIST= ', np.min(psame), 'Same points MAX_DIST= ', np.max(psame))
        print('Not Same Points Mean= ', np.mean(pnsame), 'Not same Points STDEV= ', np.std(pnsame), 'Not Same Poinst MIN_DIST= ', np.min(pnsame), 'Not Same Points MAX_DIST= ', np.max(pnsame))

        dm = (np.std(psame) + np.std(pnsame)) / 2
        nm = abs(np.mean(psame) - np.mean(pnsame))

        d = nm / dm
        print(d)
        return d


    def run(self):
        for layer in self.HIDDEN_LAYERS:
            for loss in self.LOSSES:
                for opt in self.OPTIMIZERS:
                    for act in self.ACTIVATION:
                        for epoch in self.EPOCHS:
                            self.hyper_parameters.append([layer, loss, opt, act, epoch])

        for i in range(500):
            p = random.choice(self.hyper_parameters)
            layer, loss, opt, act, epoch = p[0], p[1], p[2], p[3], p[4]
            self.process(layer, loss, opt, act, epoch)

    def define_args(self, parser):
        parser.add_argument('train_d', help='training data file')
        parser.add_argument('test_d', help='test data fiel')
        parser.add_argument('outf', help='output file')

if __name__ == '__main__':
    GridScript().start()
