import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, concatenate
from keras.optimizers import RMSprop
from keras import backend as K
from keras import regularizers
import keras.optimizers as optimizers
import tensorflow as tf

from diskarray import DiskArray
from basescript import BaseScript

from utils import create_base_network, triplet_loss, lossless_triplet_loss, batch_hard_triplet_loss

class TripletNetwork(BaseScript):
    INPUT_DIM = 300

    def run(self):

        self.train = DiskArray(self.args.train_d, dtype=self._get_dtype())
        self.test = DiskArray(self.args.test_d, dtype=self._get_dtype())
        self.alpha = self.args.alpha

        model = self.init_model()
        self.compile_model(model)
        self.train_model(model)
        self.test_model(model)

        model.save(self.args.model)

    def _get_dtype(self):
        d = self.INPUT_DIM

        return [('anchor', np.float32, d), ('positive', np.float32, d), ('negative', np.float32, d)]

    def init_model(self):
        base_network = create_base_network(self.INPUT_DIM)

        anchor_in = Input(shape=(self.INPUT_DIM,))
        positive_in = Input(shape=(self.INPUT_DIM,))
        negative_in = Input(shape=(self.INPUT_DIM,))

        anchor_out = base_network(anchor_in)
        positive_out = base_network(positive_in)
        negative_out = base_network(negative_in)

        merged_vector = concatenate([anchor_out, positive_out, negative_out], axis=1)

        model = Model(inputs=[anchor_in, positive_in, negative_in], outputs=merged_vector)

        return model

    def compile_model(self, model):
        model.compile(loss=triplet_loss,
                      metrics=['mse'],
                      optimizer=optimizers.Adam(),
                      )

    def train_model(self, model):
        T = self.train
        n = 3500
        train_data = [T['anchor'][:n], T['positive'][:n], T['negative'][:n]]
        model.fit(train_data[:], np.zeros(n),
                epochs=self.args.epoch, shuffle=True, steps_per_epoch=None, batch_size=128,
        )
        model.save(self.args.model)

    def test_model(self, model):
        labels = np.zeros(len(self.test[:]))
        test_loss, test_mse = model.evaluate(
                        [
                            self.test['anchor'], self.test['positive'], self.test['negative'],
                        ],
                        labels,
                    )

        print("Test Loss = ", test_loss)
        print("Test Mse = ", test_mse)

    def define_args(self, parser):
        parser.add_argument('train_d', help='training file')
        parser.add_argument('test_d', help='testing file')
        parser.add_argument('model', help='model path with name')
        parser.add_argument('epoch', type=int, help='no. of epochs for training')
        parser.add_argument('alpha', type=float, help='margin value')

if __name__ == '__main__':
    TripletNetwork().start()
