import os
import numpy as np
import requests

from diskarray import DiskArray
from wordvecspace import WordVecSpaceMem
from basescript import BaseScript

class CorrectionalTraining(BaseScript):
    VEC_DIM = 300
    LABELS = [0,1]

    def __init__(self):
        super(CorrectionalTraining, self).__init__()
        self.train_f = DiskArray(self.args.train_f, shape=(self.get_shape(),), dtype=self.get_dtype())
        self.wv = WordVecSpaceMem(self.args.wvspace_f)

    def get_user_token(self):
        token = input("Enter the search token: ")

        return token

    def get_shape(self):
        if not os.path.exists(self.args.train_f):
            return 0

        dtype = self.get_dtype()
        shape = os.stat(self.args.train_f).st_size // np.dtype(dtype).itemsize
        return shape

    def get_nearest_token(self, token):
        url = 'http://dev0.servers.deepcompute.com:8888/api/v1/get_k_nearest_cosine?word={}&k=10'.format(token)
        #url = 'http://dev0.servers.deepcompute.com:8888/api/v1/get_nn_model_k_nearest?word={}&k=10'.format(token)
        response = requests.get(url)
        response = response.json()
        result = response.get('result')

        return result

    def get_user_label(self, token, nearest_token):
        #name = nearest_token.get('name', '')
        #nearest_token = nearest_token.get('word2', '')
        name = token
        '''
        if not name:
            name = nearest_token
        '''
        print('the nearest token is %s' % name)
        label = input("Mark the distance between {} and {}: ".format(token, nearest_token))

        return int(label)

    def get_token_vector(self, token, nearest_token):
        token_vec = self.wv.get_word_vector(token)
        nearest_tok_vec = self.wv.get_word_vector(nearest_token)

        return token_vec, nearest_tok_vec

    def append_label_to_diskarray(self, vec1, vec2, word1, word2, label):
        self.train_f.append((vec1, vec2, word1, word2, label))

    def get_dtype(self):
        return [('vec1', np.float32, self.VEC_DIM),
                ('vec2', np.float32, self.VEC_DIM),
                ('word1', 'S', self.VEC_DIM),
                ('word2', 'S', self.VEC_DIM),
                ('label', np.int),
            ]

    def run(self):
        try:
            while True:
                token = self.get_user_token()
                nearest_tokens = self.get_nearest_token(token)
                for nearest_token in nearest_tokens:
                    label = int(self.get_user_label(token, nearest_token))
                    if label not in self.LABELS:
                        continue
                    vec1, vec2 = self.get_token_vector(token, nearest_token)
                    self.append_label_to_diskarray(vec1, vec2, token, nearest_token, label)
        finally:
            self.train_f.flush()

    def define_args(self, parser):
        parser.add_argument('train_f', help='diskarray train file')
        parser.add_argument('wvspace_f', help='wvspace file')

if __name__=="__main__":
    CorrectionalTraining().start()
