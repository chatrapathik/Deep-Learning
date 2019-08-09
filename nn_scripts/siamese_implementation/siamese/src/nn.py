import csv

import numpy as np
import random as rn

from scipy.spatial.distance import euclidean as e

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, concatenate, BatchNormalization
from keras.optimizers import RMSprop
from keras import backend as K
from keras import regularizers
import keras.optimizers as optimizers
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

from diskarray import DiskArray
from basescript import BaseScript

class SiameseNetwork(BaseScript):
    INPUT_DIM = 300

    def run(self):
        self.train_d = DiskArray(self.args.trainf, dtype=self._get_dtype())
        self.test_d = DiskArray(self.args.testf, dtype=self._get_dtype())

        model = self.init_model()
        self.compile_model(model)
        self.train_model(model)
        self.test_model(model)
        self.save_model(model)

    def _euclidean_dis_loss(self, y_true, y_pred):
        return K.sqrt(K.sum(K.square(y_pred - y_true), axis=1))

    def _get_dtype(self):
        d = self.INPUT_DIM
        return [('vec1', np.float32, d), ('vec2', np.float32, d), ('label', np.int)]

    def _euclidean_distance(self, vects):
        x, y = vects
        return K.sum(K.square(x - y), axis=1, keepdims=True)

    def _mean_squared_layer(self, vects):
        x, y = vects

        return K.mean(K.square(x - y), axis=-1)

    def _dist_output_shape(self, shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)

    def _contrastive_loss(self, y_true, y_pred):
       margin = 1

       return K.mean(
                (y_true * K.square(y_pred)) +
                (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


    def _create_base_network(self):
        a = 'tanh'
        ar = regularizers.L1L2()
        model = Sequential()
        model.add(Dense(self.INPUT_DIM, input_shape=(self.INPUT_DIM, ), activation=a))
        model.add(Dense(600, activation=a))
        model.add(Dense(self.INPUT_DIM, activation=a))
        return model

    def init_model(self):
        base_network = self._create_base_network()

        input_a = Input(shape=(self.INPUT_DIM,))
        input_b = Input(shape=(self.INPUT_DIM,))

        # because we re-use the same instance `base_network`,
        # the weights of the network
        # will be shared across the two branches
        processed_a = base_network(input_a)
        processed_b = base_network(input_b)

        distance = Lambda(self._euclidean_distance, output_shape=self._dist_output_shape)([processed_a, processed_b])

        model = Model(inputs=[input_a, input_b], outputs=distance)

        return model

    # train
    def compile_model(self, model):
        model.compile(loss=self._shivam_loss,
                      metrics=[self._euclidean_dis_loss],
                      optimizer=optimizers.Adam()
                      )

    def train_model(self, model):
        #checkpoint = [ModelCheckpoint(self.args.model, monitor='val_loss', verbose=1, save_best_only=True, mode='min')]
        history = model.fit([self.train_d['vec1'], self.train_d['vec2']], self.train_d['label'], validation_split=0.20, batch_size=128, epochs=self.args.epochs, shuffle=True)

        plt.plot(history.history['loss'], label='training loss on 80% training')
        #plt.plot(history.history['val_loss'], label='validation loss on 20% training')
        plt.legend()

        plt.savefig(self.args.loss_graph)

    # compute final accuracy on training and test sets
    def test_model(self, model):
        test_loss, ed_loss= model.evaluate([self.test_d['vec1'], self.test_d['vec2']], self.test_d['label'])

        print("Test Loss = ", test_loss)
        print('Test ed_loss =', ed_loss)

        psame = []
        pnsame = []

        csv_f = open(self.args.csv, 'w')
        csv_file = csv.writer(csv_f)
        csv_file.writerow(['label', 'prediction'])

        for i in range(len(self.test_d['vec1'])):
            vec1 = self.test_d['vec1'][i]
            vec2 = self.test_d['vec2'][i]

            r_vec1 = vec1.reshape(1, len(vec1))
            r_vec2 = vec2.reshape(1, len(vec2))

            pred_val = model.predict([r_vec1, r_vec2])[0]

            label = self.test_d['label'][i]

            if label == 0:
                psame.append(pred_val)
            else:
                pnsame.append(pred_val)

            csv_file.writerow([label, pred_val])

        print('Same Points Mean= ', np.mean(psame), 'Same Points STDEV= ', np.std(psame), 'Same Poinst MIN_DIST= ', np.min(psame), 'Same points MAX_DIST= ', np.max(psame))
        print('Not Same Points Mean= ', np.mean(pnsame), 'Not same Points STDEV= ', np.std(pnsame), 'Not Same Poinst MIN_DIST= ', np.min(pnsame), 'Not Same Points MAX_DIST= ', np.max(pnsame))

        dm = (np.std(psame) + np.std(pnsame)) / 2
        nm = abs(np.mean(psame) - np.mean(pnsame))

        d = nm / dm
        print(d)

        yticks = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        plt.hist(psame, bins=50, alpha=0.5, label='same points')
        plt.hist(pnsame, bins=50, alpha=0.5, label='not same points')
        plt.legend(loc='upper right')
        plt.yticks(yticks)
        plt.savefig(self.args.image)

    def _remove_layers(self, model):
        model.layers.pop(-1)

    def save_model(self, model):
        model.save(self.args.model)
        print(model.summary())

    def define_args(self, parser):
        parser.add_argument('trainf', help='training file')
        parser.add_argument('testf',  help='testing file')
        parser.add_argument('image',  help='image name')
        parser.add_argument('csv', help='csv file name')
        parser.add_argument('epochs',  help='testing file', type=int)
        parser.add_argument('model', help='model path with name')
        parser.add_argument('loss_graph', help='loss graph during training')

if __name__ == '__main__':
    SiameseNetwork().start()
