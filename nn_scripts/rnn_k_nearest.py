import sys
import random

import numpy as np

from diskarray import DiskArray
from diskdict import DiskDict

inp_f = sys.argv[1]
dict_file = sys.argv[2]

index = random.randint(0, 866344)
print(index)
k = int(sys.argv[3])

def test():
    d = DiskArray(inp_f, dtype=[('vec', np.float32, 128)])
    mapping = DiskDict(dict_file)

    print('The given word is', mapping[str(index)])
    vectors = d['vec']
    vec = vectors[index].reshape(1, len(vectors[0]))
    vectors_t = vectors.T

    dists = np.dot(vec, vectors_t)
    k_near = np.argsort(dists)[0]

    words = []
    for i in k_near:
        words.append(mapping[str(i)])

    return words

words = test()
print(words[:k])
