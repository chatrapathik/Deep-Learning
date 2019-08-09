import os
import pickle
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Input, SimpleRNN,  LSTM
from keras.utils import to_categorical

from diskarray import DiskArray
from basescript import BaseScript

class StringEmbeddingsScript(BaseScript):
    CHAR_NONE = '\x00'
    CHAR_START = '\x01'
    CHAR_END = '\x02'

    def create_model(self, num_units, word_len, num_unique_chars):
        input_shape = (word_len, num_unique_chars)

        model = Sequential()
        model.add(LSTM(num_units, input_shape=input_shape, unroll=True))
        model.add(Dense(num_unique_chars, activation='softmax'))

        model.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['mse'])
        return model

    def load_data(self, fpath):
        words = [line[:-1] for line in open(fpath)]
        max_len = max(len(w) for w in words) + 2 # adding 2 for start and end chars
        nwords = len(words)

        chars = list(sorted(list(set(list(''.join(words))))))
        chars = [self.CHAR_NONE, self.CHAR_START, self.CHAR_END] + chars
        charmap = { c: i for i, c in enumerate(chars) }
        nchars = len(chars)
        char_none = to_categorical(charmap[self.CHAR_NONE], num_classes=nchars)

        data = DiskArray('866k_training_data.array', shape=(nwords, max_len, nchars), dtype=np.float32)
        labels = DiskArraay('866k_labels_data.array', shape=(nwords, nchars), dtype=np.float32)

        for i in range(nwords):
            w = words[i][:-1]
            last_char = words[i][-1]

            w = '%s%s%s' % (self.CHAR_START, w, self.CHAR_END)
            w = [to_categorical(charmap[x], num_classes=nchars) for x in w]
            w = w + ([char_none] * (max_len - len(w)))

            data[i] = w
            labels[i] = to_categorical(charmap[last_char], num_classes=nchars)

        data.flush()
        labels.flush()
        return data, labels

    def run(self):
        fpath = self.args.text
        fpath_pickled = self.args.text + ".pickled"

        
        _, _ = self.load_data(fpath)
        import pdb;pdb.set_trace()
        if not os.path.exists(fpath_pickled):
            data, labels = self.load_data(fpath)
            pickle.dump((data, labels), open(fpath_pickled, 'wb'))
        else:
            data, labels = pickle.load(open(fpath_pickled, 'rb'))

        nwords, max_len, nchars = data.shape

        model = self.create_model(128, max_len, nchars)
        history = model.fit(data, labels, epochs=self.args.epochs, batch_size=512)
        plt.plot(history.history['loss'])
        plt.savefig(self.args.loss_image)
        model.save(self.args.model_name)

    def define_args(self, parser):
        parser.add_argument('text', help='text input file')
        parser.add_argument('epochs', type=int, help='num of epochs')
        parser.add_argument('loss_image', help='loss image name')
        parser.add_argument('model_name', help='model name to save')

if __name__ == '__main__':
    StringEmbeddingsScript().start()
