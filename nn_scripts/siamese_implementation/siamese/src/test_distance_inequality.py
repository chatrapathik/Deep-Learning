import sys
import csv
import random

import numpy as np
from scipy.spatial import distance
from wordvecspace import WordVecSpaceMem
from keras import backend as K
from keras.models import load_model

def _euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def _cosine_distance(vects):
    x, y = vects
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)

def _dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape2[0], 1)

def _reshape(vec):
    return vec.reshape(1, len(vec))

def get_diff_vec(v1, v2):
    diff_vector = np.zeros(len(v1))
    diff_vector[0:len(v1)] = np.abs(v1 - v2)
    diff_vector = np.insert(diff_vector, len(diff_vector), distance.cosine(v1, v2))
    diff_vector = diff_vector.reshape(1, len(diff_vector))

    return diff_vector

def test(inpf, model, outf):
    wv = WordVecSpaceMem(inpf)
    model = load_model(model, custom_objects=dict(
                            _euclidean_distance=_euclidean_distance,
                            _dist_output_shape = _dist_output_shape)
                        )

    inequality_count = 0

    for i in range(1000):
        index1, index2, index3 = random.sample(range(wv.nvecs), 3)

        vec1 = wv.get_word_vector(index1)
        vec2 = wv.get_word_vector(index2)
        vec3 = wv.get_word_vector(index3)

        vec1 = _reshape(vec1)
        vec2 = _reshape(vec2)
        vec3 = _reshape(vec3)
        dist_v13 = model.predict([vec1, vec3])
        dist_v12 = model.predict([vec1, vec2])
        dist_v23 = model.predict([vec2, vec3])

        '''
        diff_vec12 = get_diff_vec(vec1, vec2)
        diff_vec13 = get_diff_vec(vec1, vec3)
        diff_vec23 = get_diff_vec(vec2, vec3)

        dist_v13 = 1 - model.predict(diff_vec13)[0][0]
        dist_v12 = 1 - model.predict(diff_vec12)[0][0]
        dist_v23 = 1 - model.predict(diff_vec23)[0][0]
        '''

        is_inequality = dist_v13 <= (dist_v12 + dist_v23)
        outf.writerow([index1, index2, index3, dist_v13, dist_v12, dist_v23, is_inequality])
        if not is_inequality:
            inequality_count += 1

    print(inequality_count)

if __name__ == '__main__':
    inpf = sys.argv[1]
    model = sys.argv[2]
    f = open(sys.argv[3], 'w')
    outf = csv.writer(f)
    outf.writerow(['index1', 'index2', 'index3', 'dist_v13', 'dist_v12', 'dist_v23', 'is_inequality'])
    test(inpf, model, outf)
