import sys

import numpy as np
from keras.models import load_model
from keras import backend as K

from wordvecspace import WordVecSpaceMem
from diskarray import DiskArray


def _euclidean_dis_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=0))

model = load_model(sys.argv[1], custom_objects=dict(_euclidean_dis_loss=_euclidean_dis_loss))
out_f = DiskArray(sys.argv[2], shape=(0,), dtype=[('vec', np.float32, 300)])

wv = WordVecSpaceMem(sys.argv[3])

def get_tras_vectors():
    for i in range(wv.nvecs):
        word = wv.get_word_at_index(i)
        vec = wv.get_word_vector(word, raise_exc=True)
        vec = vec.reshape(1, 300)
        t_vec = model.predict(vec)
        out_f.append((t_vec[0], ))

get_tras_vectors()
out_f.flush()
