import csv

import numpy as np
import random as rn

from scipy.spatial.distance import euclidean as e

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, concatenate, BatchNormalization
from keras.optimizers import RMSprop
from keras import backend as K
from keras import regularizers
import keras.optimizers as optimizers
from keras.models import load_model

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

    def _get_dtype(self):
        d = self.INPUT_DIM
        return [('vec1', np.float32, d), ('vec2', np.float32, d), ('label', np.int)]

    def _euclidean_distance(self, vects):
        x, y = vects
        return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

    def _cosine_distance(self, vects):
        x, y = vects
        x = K.l2_normalize(x, axis=-1)
        y = K.l2_normalize(y, axis=-1)
        return -K.mean(x * y, axis=-1, keepdims=True)

    def _dist_output_shape(self, shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)

    def contrastive_loss(self, y_true, y_pred):
        '''Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        '''
        margin = 1
        return K.mean(
                (y_true * K.square(y_pred)) +
                (1 - y_true) * K.square(K.maximum(margin - y_pred, 0))
        )

    def _create_base_network(self, input_dim):
        a = 'tanh'
        ar = regularizers.L1L2()
        network = Sequential()
        network.add(Dense(300, input_shape=(input_dim,), activation=a,activity_regularizer=ar))
        network.add(Dense(150, activation=a))
        network.add(Dense(250, activation=a))
        network.add(Dense(300, activation=a))
        return network

    def init_model(self):
        base_network = self._create_base_network(self.INPUT_DIM)

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
        model.compile(loss='mse',
                      metrics=['mse'],
                      optimizer=optimizers.Adam(lr=.001, beta_1=0.002, beta_2=0.002)
                      )

    def train_model(self, model):
        model.fit([self.train_d['vec1'], self.train_d['vec2']], self.train_d['label'], batch_size=128, epochs=self.args.epochs, shuffle=True)

    # compute final accuracy on training and test sets
    def test_model(self, model):
        test_loss, test_mse= model.evaluate([self.test_d['vec1'], self.test_d['vec2']], self.test_d['label'])

        print("Test Loss = ", test_loss)
        print("Test Mse = ", test_mse)

        psame = []
        pnsame = []
        dist = []
        e_dist = []

        csv_f = open('test.csv', 'w')
        csv_file = csv.writer(csv_f)
        csv_file.writerow(['label', 'prediction'])

        for i in range(len(self.test_d['vec1'])):
            vec1 = self.test_d['vec1'][i]
            vec2 = self.test_d['vec2'][i]

            r_vec1 = vec1.reshape(1, len(vec1))
            r_vec2 = vec2.reshape(1, len(vec2))


            #pred_val = model.predict([r_vec1, r_vec2])[0][0]

 
            pred_val = e(r_vec1, r_vec2)
            e_dist.append(pred_val)


            dist.append(pred_val)
            label = self.test_d['label'][i]

            if label == 0:
                psame.append(pred_val)
            else:
                pnsame.append(pred_val)

            csv_file.writerow([label, pred_val])

        '''
        print('Same Points Mean= ', np.mean(psame), 'Same Points STDEV= ', np.std(psame), 'Same Poinst MIN_DIST= ', np.min(psame), 'Same points MAX_DIST= ', np.max(psame))
        print('Not Same Points Mean= ', np.mean(pnsame), 'Same Points STDEV= ', np.std(pnsame), 'Same Poinst MIN_DIST= ', np.min(pnsame), 'Same Points MAX_DIST= ', np.max(pnsame))
        '''

        dm = (np.std(psame) + np.std(pnsame)) / 2
        nm = abs(np.mean(psame) - np.mean(pnsame))

        d = nm / dm
        print(d)
        #return d

        yticks = [50, 100, 150, 200, 250, 300]
        plt.hist(psame, bins=30, alpha=0.5, label='same points', color='orange')
        plt.hist(pnsame, bins=30, alpha=0.5, label='not same points', color='yellow')
        #plt.hist(e_dist, bins=50, alpha=0.5, label='z')
        plt.yticks(yticks)
        plt.legend(loc='upper right')
        plt.savefig('./shivam/unsupervised_same_and_not_same.png')

    '''
        self.plot_hist(psame, './images/out_same_points_1_t.png')
        self.plot_hist(pnsame, './images/out_not_same_points_1_.png')
        self.plot_hist(e_dist, './images/out_all_points_1_t.png')


    def plot_hist(self, dist, name):
        yticks = [50, 100, 150, 200, 250, 300]
        plt.hist(dist, bins=40)
        plt.yticks(yticks)
        plt.savefig(name)
    '''

    def _remove_layers(self, model):
        model.layers.pop(-1)

    def save_model(self, model):
        model.save(self.args.model)

    def define_args(self, parser):
        parser.add_argument('trainf', help='training file')
        parser.add_argument('testf',  help='testing file')
        parser.add_argument('epochs',  help='testing file', type=int)
        parser.add_argument('model', help='model path with name')

if __name__ == '__main__':
    SiameseNetwork().start()
