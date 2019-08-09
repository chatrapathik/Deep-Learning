import numpy as np

from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Input, Lambda, concatenate
from keras.optimizers import RMSprop
from keras import backend as K
from keras import regularizers
import keras.optimizers as optimizers
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from diskarray import DiskArray
from basescript import BaseScript

from utils import create_base_network, triplet_loss

class DistanceFunction(BaseScript):
    INPUT_DIM = 300

    def __init__(self):
        super(DistanceFunction, self).__init__()
        self.test_d = DiskArray(self.args.testf, dtype=self._get_dtype())

    def _get_dtype(self):
        d = self.INPUT_DIM
        return [('vec1', np.float32, d), ('vec2', np.float32, d), ('label', np.int)]

    def run(self):
        existing_model = load_model(self.args.model_f, custom_objects=dict(triplet_loss=triplet_loss))
        existing_weights = existing_model.get_weights()

        network = create_base_network(self.INPUT_DIM)
        #model = self.compile_model(network)


        #w = [np.random.uniform(low=-1.0, high=1.0, size=x.shape) for x in existing_weights]
        network.set_weights(existing_weights)

        self.test_model(network)


    def compile_model(self, model):
        model.compile(loss='mae',
                      optimizer=optimizers.Adam(),
                      )

        return model

    def ed_dist(self, vec1,vec2):
        return np.sqrt(np.sum((vec1-vec2)**2, axis=1))

    def test_model(self, model):
        psame = []
        npsame = []

        labels = self.test_d['label']
        for i in range(len(labels)):
            vec1 = self.test_d['vec1'][i]
            vec2 = self.test_d['vec2'][i]
            tvec1 = model.predict(vec1.reshape(1,300))
            tvec2 = model.predict(vec2.reshape(1,300))
            '''

            tvec1 = vec1.reshape(1,300)
            tvec2 = vec2.reshape(1,300)
            '''
            dist = self.ed_dist(tvec1, tvec2)

            if labels[i] == 0:
                psame.append(dist)

            else:
                npsame.append(dist)

        smean = np.mean(psame)
        nsmean = np.mean(npsame)

        s_std = np.std(psame)
        ns_std = np.std(npsame)

        d = abs((smean - nsmean) / ((s_std + ns_std) / 2))

        print('dup\'s mean: ', smean,
              'dup\'s min: ', np.min(psame),
              'dup\'s max: ', np.max(psame),
              'dup\'s stdev: ', s_std,
             )


        print('non-dup\'s mean: ', nsmean,
              'non-dup\'s min: ', np.min(npsame),
              'non-dup\'s max: ', np.max(npsame),
              'non-dup\'s stdev: ', ns_std,
             )

        print('Cohen\'s D: ', d)

        self.generate_histogram(psame, npsame)

    def generate_histogram(self, psame, npsame):
        same_dist = []
        not_same_dist = []
        for i in range(len(psame)):
            same_dist.append(psame[i][0])

        for i in range(len(npsame)):
            not_same_dist.append(npsame[i][0])

        print (len(same_dist))
        print (len(not_same_dist))

        #plt.hist(same_dist, bins=40, alpha=0.5, label='same-points', color='black')
        plt.hist(not_same_dist, bins=40, alpha=0.5, label='not-same-points', color='thistle')
        plt.title(self.args.title)
        plt.legend(loc='upper right')
        plt.savefig(self.args.hist)


    def define_args(self, parser):
        parser.add_argument('testf', help='testing file containing untransformed vectors')
        parser.add_argument('model_f', help='existing model')
        parser.add_argument('margin', type=float, help='margin used in the existing model')
        parser.add_argument('hist', help='histogram name')
        parser.add_argument('title', help='title of the histogram')

if __name__ == "__main__":
    DistanceFunction().start()
