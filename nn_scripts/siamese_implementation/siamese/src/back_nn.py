import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, concatenate
from keras.optimizers import RMSprop
from keras import backend as K

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
        network = Sequential()
        network.add(Dense(300, input_shape=(input_dim,), activation='tanh'))
        network.add(Dropout(0.1))
        network.add(Dense(200, input_shape=(input_dim,), activation='tanh'))
        network.add(Dropout(0.1))
        network.add(Dense(100, input_shape=(input_dim,), activation='tanh'))
        network.add(Dropout(0.1))
        network.add(Dense(50, input_shape=(input_dim,), activation='tanh'))
        network.add(Dropout(0.1))
        network.add(Dense(25, input_shape=(input_dim,), activation='tanh'))
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

        distance = Lambda(self._cosine_distance, output_shape=self._dist_output_shape)([processed_a, processed_b])

        model = Model(inputs=[input_a, input_b], outputs=distance)

        return model

    # train
    def compile_model(self, model):
        model.compile(loss='mean_absolute_error',
                      metrics=['mae'],
                      optimizer='rmsprop',
                      )

    def train_model(self, model):
        model.fit([self.train_d['vec1'], self.train_d['vec2']], self.train_d['label'], epochs=20, shuffle=True)

    # compute final accuracy on training and test sets
    def test_model(self, model):
        test_loss, test_mae= model.evaluate([self.test_d['vec1'], self.test_d['vec2']], self.test_d['label'])

        print("Test Loss = ", test_loss)
        print("Test Mae = ", test_mae)

    def _remove_layers(self, model):
        model.layers.pop(-1)

    def save_model(self, model):
        #self._remove_layers(model)
        #new_model = model.layers[-2]
        #new_model.save(self.args.model)
        #from keras.utils import plot_model
        #plot_model(model, to_file='data/model.png', show_shapes=True)
        model.save(self.args.model)
        from keras.models import load_model


        #load_model(self.args.model,
        #    custom_objects=dict(
        #        _euclidean_distance=self._euclidean_distance,
        #        _eucl_dist_output_shape=self._eucl_dist_output_shape)
        #)

    def define_args(self, parser):
        parser.add_argument('trainf', help='training file')
        parser.add_argument('testf',  help='testing file')
        parser.add_argument('model', help='model path with name')

if __name__ == '__main__':
    SiameseNetwork().start()
