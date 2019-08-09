import os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))

from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
from keras.utils import to_categorical
from keras import optimizers

from basescript import BaseScript
from diskarray import DiskArray

class StringEmbeddingsScript(BaseScript):
    CHAR_NONE = '\x00'
    CHAR_START = '\x01'
    CHAR_END = '\x02'

    def create_model(self, num_units, word_len, num_unique_chars):
        input_shape = (word_len, num_unique_chars)

        model = Sequential()
        model.add(LSTM(num_units, input_shape=input_shape, unroll=True))
        model.add(Dense(num_unique_chars, activation='softmax'))

        model.compile(optimizer=optimizers.Adam(lr=0.003),
                        loss='categorical_crossentropy',
                        metrics=['mse'])
        return model

    def get_char_to_int(self, fpath):
        words = [line[:-1] for line in open(fpath)]
        max_len = max(len(w) for w in words) + 2 # adding 2 for start and end chars
        nwords = len(words)

        chars = list(sorted(list(set(list(''.join(words))))))
        chars = [self.CHAR_NONE, self.CHAR_START, self.CHAR_END] + chars
        charmap = { c: i for i, c in enumerate(chars) }
        nchars = len(chars)

        return max_len, nchars, nwords, words, charmap

    def load_data(self, max_len, nchars, nwords, words, charmap):
        char_none = to_categorical(charmap[self.CHAR_NONE], num_classes=nchars)
        data = np.zeros(shape=(nwords, max_len, nchars), dtype=np.float32)
        labels = np.zeros(shape=(nwords, nchars), dtype=np.float32)

        for i in range(nwords):
            w = words[i][:-1]
            last_char = words[i][-1]
            w = '%s%s%s' % (self.CHAR_START, w, self.CHAR_END)
            w = [to_categorical(charmap[x], num_classes=nchars) for x in w]
            w = w + ([char_none] * (max_len - len(w)))
            data[i] = w
            labels[i] = to_categorical(charmap[last_char], num_classes=nchars)

        return data, labels

    def generator(self, max_len, nchars, nwords, words, charmap, b_size):
        while 1:
            char_none = to_categorical(charmap[self.CHAR_NONE], num_classes=nchars)
            num_batches = (nwords//b_size) + 1

            for i in range(num_batches):
                start = b_size * i
                end = b_size * (i + 1)

                split_words = words[start:end]
                n = len(split_words)

                data = np.zeros(shape=(n, max_len, nchars), dtype=np.float32)
                labels = np.zeros(shape=(n, nchars), dtype=np.float32)

                for i in range(n):
                    w = split_words[i][:-1]
                    last_char = split_words[i][-1]
                    w = '%s%s%s' % (self.CHAR_START, w, self.CHAR_END)
                    w = [to_categorical(charmap[x], num_classes=nchars) for x in w]
                    w = w + ([char_none] * (max_len - len(w)))
                    data[i] = w
                    labels[i] = to_categorical(charmap[last_char], num_classes=nchars)

                yield data, labels

    def run(self):

        fpath = self.args.text
        fpath_pickled = self.args.text + ".pickled"

        max_len, nchars, nwords, words, charmap = self.get_char_to_int(fpath)

        disk_array = DiskArray(self.args.out_f, shape=(0,), dtype=[('vec', np.float32, 128)])
        '''
        if not os.path.exists(fpath_pickled):
            data, labels = self.load_data(max_len, nchars, nwords, words, charmap)
            pickle.dump((data, labels), open(fpath_pickled, 'wb'))
        else:
            data, labels = pickle.load(open(fpath_pickled, 'rb'))
        '''

        if not os.path.exists(self.args.model_name):
            model = self.create_model(128, max_len, nchars)

            #history = model.fit(data, labels, epochs=self.args.epochs, batch_size=128)
            generator = self.generator(max_len, nchars, nwords, words, charmap, 2048)
            model.fit_generator(generator, steps_per_epoch= nwords/2048, epochs=self.args.epochs)
        else:
            model = load_model(self.args.model_name)

        model.save(self.args.model_name)

        if self.args.layer == 'lstm_layer':
            self.log.info('Accessing the layer weights')
            new_model = Sequential()
            new_model.add(LSTM(128, input_shape=(max_len, nchars), unroll=True))
            weights = model.layers[0].get_weights()
            new_model.set_weights(weights)
            model_p = new_model
        else:
            model_p = model

        self.log.info('started predicting')
        for word in words:
            test_data, test_lables = self.load_data(max_len, nchars, 1, [word], charmap)
            p_out = model_p.predict(test_data)
            disk_array.append((p_out[0],))

        disk_array.flush()

    def define_args(self, parser):
        parser.add_argument('text', help='text input file')
        parser.add_argument('epochs', type=int, help='num of epochs')
        parser.add_argument('model_name', help='model name to save')
        parser.add_argument('layer', help='Use lstm layer or output layer')
        parser.add_argument('out_f', help='out_f name')

if __name__ == '__main__':
    StringEmbeddingsScript().start()
