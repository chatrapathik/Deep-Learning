import json

import random
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout
from keras import backend as k
from keras import regularizers
import keras.optimizers as optimizers
from keras import backend as K

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from basescript import  BaseScript
from diskarray import DiskArray as D

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))

class Dummy(BaseScript):
    INPUT_DIM = 300

    def run(self):
        self.train_d = D(self.args.trainf, dtype=self._get_dtype())
        self.test_d = D(self.args.testf, dtype=self._get_dtype())

        model = self.init_model()
        self.compile_model(model)
        self.train_model(model)
        self.test_model(model)

        model.save(self.args.model)

    def _get_dtype(self):
        return [('vec1', np.float32, 300), ('vec2', np.float32, 300), ('label', np.int, 1)]
        #return [('vecs', np.float32, 300)]

    def get_train_test(self, train, test):
        tr = json.loads(train.read())
        te = json.loads(test.read())

        return np.array(tr, dtype=np.float32), np.array(te, dtype=np.float32)

    def _euclidean_dis_loss(self, y_true, y_pred):
        return K.sqrt(K.sum(K.square(y_pred - y_true), axis=0, keepdims=True))

    def init_model(self):
        a = 'tanh'
        s = 'softsign'
        l = 'linear'
        '''
        input_vec = Input(shape=(self.INPUT_DIM,))
        model = Dense(self.INPUT_DIM, activation=a)(input_vec)
        model = Model(input_vec, model)

        '''
        # simple network

        model = Sequential()
        model.add(Dense(self.INPUT_DIM, input_shape=(self.INPUT_DIM, ), activation=a))
        model.add(Dense(250, activation=a))
        model.add(Dense(150, activation=a))
        model.add(Dense(self.INPUT_DIM, activation=a))

        return model


        # complex network
        '''
        model = Sequential()

        model.add(Dense(self.INPUT_DIM, input_shape=(self.INPUT_DIM, ), activation=a))
        model.add(Dense(600, activation=a))
        model.add(Dense(600, activation=a))
        model.add(Dense(900, activation=a))
        model.add(Dense(900, activation=a))
        model.add(Dense(3000, activation=a))
        model.add(Dense(900, activation=a))
        model.add(Dense(900, activation=a))
        model.add(Dense(600, activation=a))
        model.add(Dense(600, activation=a))
        model.add(Dense(self.INPUT_DIM, activation=a))

        return model
        '''
    def train_model(self, model):
        history = model.fit([self.train_d['vec1']], self.train_d['vec1'], batch_size=1024, epochs=self.args.epochs, shuffle=True)
        #history = model.fit([self.train_d['vecs']], self.train_d['vecs'], batch_size=1024, epochs=self.args.epochs, shuffle=True)
        plt.plot(history.history['loss'])
        plt.savefig(self.args.loss_history)

    def compile_model(self, model):
        model.compile(loss='mse',
                        metrics=[self._euclidean_dis_loss, 'acc'],
                        optimizer=optimizers.adam(lr=0.001)
                    )

    def test_model(self, model):
        test_loss, test_mse, acc = model.evaluate([self.test_d['vec1']], self.test_d['vec1'])
        #test_loss, test_mse, acc = model.evaluate([self.test_d['vecs']], self.test_d['vecs'])

        print('the test mse loss is', test_loss)
        print('the test euc error is', test_mse)
        print('the reporetd accuracy is',  acc)

    def define_args(self, parser):
        parser.add_argument('trainf', help='train file')
        parser.add_argument('testf', help='test file')
        parser.add_argument('epochs', type=int, help='num of epoches')
        parser.add_argument('model', help='model file')
        parser.add_argument('loss_history', help='save the loss history')

if __name__ == '__main__':
    Dummy().start()
