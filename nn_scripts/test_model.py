import os
import pickle
import numpy as np

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))

from keras.models import Sequential, Model , load_model
from keras.utils import to_categorical

from basescript import BaseScript
from diskarray import DiskArray

class StringEmbeddingsScript(BaseScript):
    CHAR_NONE = '\x00'
    CHAR_START = '\x01'
    CHAR_END = '\x02'

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


    def run(self):

        fpath = self.args.text

        max_len, nchars, nwords, words, charmap = self.get_char_to_int(fpath)

        disk_array = DiskArray(self.args.out_f, shape=(0,), dtype=[('vec', np.float32, 108)])

        model = load_model(self.args.model_name)

        self.log.info('started predicting')
        for word in words:
            test_data, test_lables = self.load_data(max_len, nchars, 1, [word], charmap)
            p_out = model.predict(test_data)
            disk_array.append((p_out[0],))

        disk_array.flush()

    def define_args(self, parser):
        parser.add_argument('text', help='text input file')
        parser.add_argument('model_name', help='model name to save')
        parser.add_argument('out_f', help='out_f name')

if __name__ == '__main__':
    StringEmbeddingsScript().start()
