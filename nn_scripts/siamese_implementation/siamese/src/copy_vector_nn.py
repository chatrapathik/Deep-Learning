import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras import backend as k
from keras import regularizers

import keras.optimizers as optimizers

from basescript import  BaseScript
from diskarray import DiskArray

class Dummy(BaseScript):
    INPUT_DIM = 1

    def run(self):
        self.train_d = DiskArray(self.args.trainf, dtype=self._get_dtype())
        self.train_d = np.array([i for i in range(10000)], dtype=np.int)

        model = self.init_model()
        self.compile_model(model)
        self.train_model(model)
        self.test_model(model)
        model.save(self.args.model)

    def _get_dtype(self):
        d = self.INPUT_DIM
        return [('vec1', np.float32, d), ('vec2', np.float32, d), ('label', np.int64)]

    def init_model(self):
        a = 'tanh'
        '''
        input_vec = Input(shape=(self.INPUT_DIM,))
        model = Dense(self.INPUT_DIM, activation=a)(input_vec)
        model = Model(input_vec, model)

        '''
        model = Sequential()
        model.add(Dense(self.INPUT_DIM, input_shape=(self.INPUT_DIM, ), activation='softplus'))


        return model

    def train_model(self, model):
        #model.fit([self.train_d['vec1']], self.train_d['vec1'], epochs=self.args.epochs, batch_size=64, shuffle=True)
        model.fit([self.train_d], self.train_d, epochs=self.args.epochs, batch_size=64, shuffle=True)

    def compile_model(self, model):
        model.compile(loss='mse',
                        metrics=['mse'],
                        optimizer=optimizers.adam(lr=0.001, beta_1=0.002, beta_2=0.002)
                    )

    def test_model(self, model):
        #test_loss, test_mse = model.evaluate([self.train_d['vec2']], self.train_d['vec2'])
        test_loss, test_mse = model.evaluate([self.train_d], self.train_d)
        print('the test loss is', test_loss)
        print('the test mse is', test_mse)

        #vec = np.random.uniform(low=-1.0, high=1.0, size=self.INPUT_DIM).reshape(1,300)

        #vec = self.train_d['vec1'][0].reshape(1, 300)
        #in_num = self.train_d[567]
        in_num = np.array([11111])
        out_num = model.predict(in_num.reshape(1,))

        print(in_num)
        print(out_num)

        print(in_num == out_num)

    def define_args(self, parser):
        parser.add_argument('trainf', help='train file')
        parser.add_argument('epochs', type=int, help='num of epoches')
        parser.add_argument('model', help='model name to save')

if __name__ == '__main__':
    Dummy().start()
