import sys

from keras.Models import load_model

from wordvecspace import WordVecSpaceMem
from diskarray import DiskArray


def _euclidean_dis_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=0))

model = load_model(sys.argv[1], custom_objects=dict(_euclidean_dis_loss=_euclidean_dis_loss))
out_f = DiskArray(sys.argv[2], dtype=[('vec', np.float32, 300)])

wv = WordVecSpaceMem(sys.argv[3])

def get_tras_vectors():
    nvecs = len(wv.vectors)
    for num in range(nvecs):
        vec = wv.get_word_vector(num)
        vec = vec.reshape(1, 300)
        t_vec = model.predict(vec)
        out_f.append((t_vec, ))
