import os
import random

import numpy as np

from basescript import BaseScript
from diskarray import DiskArray
from wordvecspace import WordVecSpaceMem

from keras.models import load_model

class TrainData(BaseScript):

    VEC_DIM = 300

    def __init__(self):
        super(TrainData, self).__init__()
        self.wvspace = WordVecSpaceMem(self.args.wvspace)
        self.train_f = DiskArray(self.args.train_file, shape=(self.get_shape(),), dtype=self.get_dtype())
        self.words_f = open(self.args.words_file, 'w')
        #self.model = load_model(self.args.model)

    def get_shape(self):
        if not os.path.exists(self.args.train_f):
            return 0

        dtype = self.get_dtype()
        shape = os.stat(self.args.train_file).st_size // np.dtype(dtype).itemsize
        return shape

    def get_dtype(self):
        return [('vec1', np.float32, self.VEC_DIM),
                ('vec2', np.float32, self.VEC_DIM),
                ('label', np.int),
            ]

    def get_random_point(self):
        return random.randint(0, len(self.wvspace))

    def near_pair(self):
        index = self.get_random_point()
        word1 = self.wvspace.get_word_at_index(index)
        nearest = self.wvspace.get_nearest(word1, 10)
        n_words = self.wvspace.get_word_at_indices(nearest)
        word2 = n_words[1]
        self.add_pair(word1, word2)

    def add_pair(self, word1, word2):
        vec1 = self.wvspace.get_word_vector(word1)
        vec2 = self.wvspace.get_word_vector(word2)
        diff_vec = abs(vec1 - vec2)
        p_value = self.model.predict(vec1, vec2)
        p_value = 0 if p_value < 3 else 1
        self.train_f.append((vec1, vec2, p_value))
        self.words_f(word1 + '<====>' + word2 + '<======>' + str(p_value))

    def far_pair(self):
        index1 = self.get_random_point()
        word1 = self.wvspace.get_word_at_index(index)
        index2 = self.get_random_point()
        word2 = self.wvspace.get_word_at_index(index)
        self.add_pair(word1, word2)

    def run(self):
        for i in range(self.args.n_samples):
            word1, word2 = self.near_pair()

    def define_args(self, parser):
        parser.add_argument('train_file', metavar='training-file')
        parser.add_argument('wvspace', metavar='vector-space')
        parser.add_argument('words_file', metavar='words-file')
        parser.add_argument('n_samples', metavar='num-of-pairs')

if __name__ == '__main__':
    TrainData().start()
