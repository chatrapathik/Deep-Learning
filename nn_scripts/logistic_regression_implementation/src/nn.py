import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint

from basescript import BaseScript as BS
from diskarray import DiskArray as DA

class CohenSD(object):
    '''
    DESC:
    '''
    def __init__(self, test_d, trained_model, dim):
        self.test_d = test_d
        self.model = trained_model
        self.dim = dim

    def _get_same_diffrent_labelled(self):
        same_labelled = list()
        not_same_labelled = list()

        [
            same_labelled.append(vec['diff_vec']) if vec['label'] == 0 \
            else not_same_labelled.append(vec['diff_vec']) \
            for vec in self.test_d[:] \
        ]

        return same_labelled, not_same_labelled

    def _pred_similarity(self, same_labelled, different_labelled):
        pred_same = []
        pred_different = []

        [pred_same.append(self.model.predict(vec.reshape(1, self.dim))) for vec in same_labelled]

        [pred_different.append(self.model.predict(vec.reshape(1, self.dim))) for vec in different_labelled]

        return pred_same, pred_different

    def _calc_cohen_d(self, pred_same, pred_different):
        '''
        >>> a = 10
        >>> b = 20
        >>> _calc_cohen_d(a, b)
        >>> 30
        '''
        denominator = (np.std(pred_same) + np.std(pred_different)) / 2
        numerator = abs(np.mean(pred_same) - np.mean(pred_different))

        cohen_d = numerator / denominator

        return cohen_d

    def get_cohen_d(self):
        same_labelled, different_labelled = self._get_same_diffrent_labelled()
        pred_same, pred_different = self._pred_similarity(same_labelled, different_labelled)
        cohen_d = self._calc_cohen_d(pred_same, pred_different)

        return cohen_d

class TrainModel(BS):
    INP_DIMS = 301
    METRICS = ['acc']

    def __init__(self):
        super(TrainModel, self).__init__()
        self.train_d = DA(self.args.train_f, dtype=self._get_dtype())
        self.test_d = DA(self.args.test_f, dtype=self._get_dtype())

        self.out_act = self.args.out_act if self.args.out_act else self._gaussian_activation

    def _get_dtype(self):
        return [
                    ('diff_vec', np.float32, 301),
                    ('label', np.int, 1),
            ]

    def run(self):
        model = self.init_model()
        trained_model = self.train_model(model)
        loss, acc = self.test_model(trained_model)

        cohen_d = CohenSD(self.test_d, trained_model, self.INP_DIMS).get_cohen_d()

        print('Loss:', loss)
        print('Accuracy:', acc)
        print('Cohen\'s_d:', cohen_d)


    def init_model(self):
        model = Sequential()

        model.add(Dense(301, input_shape=(self.INP_DIMS,), activation=self.args.inp_act))
        model.add(Dense(150, activation=self.args.inp_act))
        model.add(Dense(1, activation=self.out_act))

        model.compile(
                        optimizer=self.args.optimizer,
                        loss=self.args.loss,
                        metrics=self.METRICS,
                     )

        return model

    def _gaussian_activation(self, x):
        return K.exp(-K.pow(x, 2))

    def train_model(self, model):
        checkpoint = ModelCheckpoint(self.args.model, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        model.fit(
                    self.train_d['diff_vec'], self.train_d['label'],
                    epochs=self.args.epochs,
                    shuffle=True,
                    callbacks=[checkpoint],
                    validation_split=0.05
            )

        return model

    def test_model(self, model):
        loss, acc = model.evaluate(self.test_d['diff_vec'], self.test_d['label'])

        return loss, acc

    def define_args(self, parser):
        parser.add_argument('train_f', help='training diskarray file')
        parser.add_argument('test_f', help='testing diskarray file')
        parser.add_argument('model', help='model to save')
        parser.add_argument('--loss', default='binary_crossentropy', help='loss function')
        parser.add_argument('--epochs', type=int, default=50, help='no. of epochs of training')
        parser.add_argument('--inp_act', default='softmax', help='input activation funtion')
        parser.add_argument('--out_act', help='output layer activation')
        parser.add_argument('--optimizer', default='rmsprop', help='optimizer to use')

if __name__ == '__main__':
    TrainModel().start()
