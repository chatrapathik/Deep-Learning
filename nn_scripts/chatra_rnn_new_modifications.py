import os
import pickle
import tempfile
import subprocess

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.models import Sequential, Model , load_model
from keras.layers import Dense, Input, SimpleRNN,  LSTM, Dropout
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

        model.compile(optimizer=optimizers.Adam(lr=0.002),
                        loss='categorical_crossentropy',
                        metrics=['mse'])
        return model

    def execute_cmd(self, cmd):
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        result = p.stdout.read().strip()

        return result.decode('utf-8')

    def get_char_to_int(self, fpath):
        chars_cmd = "fold -w1 {0} | sort -u".format(fpath)
        chars = self.execute_cmd(chars_cmd)
        chars = chars.split('\n')

        max_len_cmd = 'cat {0} | py -x "len(x)" | sort -n | tail -1'.format(fpath)
        max_len = self.execute_cmd(max_len_cmd)
        max_len = int(max_len) + 2 # adding 2 for start and end chars
        self.log.info('calculating max lenght is done')

        nwords_cmd = 'cat {0} | py -x "len(x)" | sort -n | wc -l'.format(fpath)
        nwords = self.execute_cmd(nwords_cmd)
        nwords = int(nwords)
        self.log.info('calculating nwords is done')

        chars = [self.CHAR_NONE, self.CHAR_START, self.CHAR_END] + chars
        charmap = { c: i for i, c in enumerate(chars) }
        nchars = len(chars)

        return max_len, nchars, nwords, charmap

    def load_data(self, max_len, nchars, nwords, charmap):
        char_none = to_categorical(charmap[self.CHAR_NONE], num_classes=nchars)
        data = DiskArray(self.args.training_data, shape=(nwords, max_len, nchars), dtype=np.float32)
        labels = DiskArray(self.args.labels_data, shape=(nwords, nchars), dtype=np.float32)

        f = open(self.args.text)
        for i, line in enumerate(f):
            line = line.strip()
            w = line[:-1]
            last_char = line[-1]
            w = '%s%s%s' % (self.CHAR_START, w, self.CHAR_END)
            w = [to_categorical(charmap[x], num_classes=nchars) for x in w]
            w = w + ([char_none] * (max_len - len(w)))
            data[i] = w
            labels[i] = to_categorical(charmap[last_char], num_classes=nchars)

        self.log.info('generating vectors is done')
        data.flush()
        labels.flush()
        return data, labels

    def get_test_data(self, max_len, nchars, nwords, words, charmap):
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

    def run(self):

        fpath = self.args.text

        max_len, nchars, nwords, charmap = self.get_char_to_int(fpath)

        disk_array = DiskArray(self.args.out_f, shape=(0,), dtype=[('vec', np.float32, 128)])
        if not os.path.exists(self.args.training_data):
            data, labels = self.load_data(max_len, nchars, nwords, charmap)
        else:
            data = DiskArray(self.args.training_data, dtype=np.float32)
            labels = DiskArray(self.args.labels_data, dtype=np.float32)

        if not os.path.exists(self.args.model_name):
            model = self.create_model(128, max_len, nchars)
            self.log.info('Started training the model')
            history = model.fit(data[:], labels[:], epochs=self.args.epochs, batch_size=128)
            plt.plot(history.history['loss'])
            plt.savefig(self.args.image_name)
        else:
            model = load_model(self.args.model_name)

        model.save(self.args.model_name)

        self.log.info('Accessing the layer weights')
        new_model = Sequential()
        new_model.add(LSTM(128, input_shape=(max_len, nchars), unroll=True))
        weights = model.layers[0].get_weights()
        new_model.set_weights(weights)

        self.log.info('started predicting')
        for word in open(fpath):
            word = word.strip()
            test_data, test_lables = self.get_test_data(max_len, nchars, 1, [word], charmap)
            p_out = new_model.predict(test_data)
            disk_array.append((p_out[0],))

        disk_array.flush()

    def define_args(self, parser):
        parser.add_argument('text', help='input text file')
        parser.add_argument('training_data', help='training file')
        parser.add_argument('labels_data', help='labels file')
        parser.add_argument('epochs', type=int, help='num of epochs')
        parser.add_argument('model_name', help='model name to save')
        parser.add_argument('image_name', help='image name')
        parser.add_argument('out_f', help='out_f name')

if __name__ == '__main__':
    StringEmbeddingsScript().start()
