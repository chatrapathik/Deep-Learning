import sys
import random

import numpy as np
import pandas as pd

from scipy.spatial import distance

from wordvecspace import WordVecSpaceMem
from diskarray import DiskArray

def k_nearest(wvspace, disk_f, word):
    wv = WordVecSpaceMem(wvspace)
    da = DiskArray(disk_f, dtype=[('vec', np.float32, 300)])
    index = wv.get_word_index(word)

    result = wv.get_nearest(index, k=10)
    print(wv.get_word_at_indices(result))

    vec = da['vec'][index].reshape(1, 300)
    vecs = da['vec']

    #dist = distance.cdist(vec, vecs, 'cosine')
    dist = distance.cdist(vec, vecs, 'euclidean')
    #dist = np.dot(vec, vecs.T)

    dist = pd.Series(dist[0])
    res = dist.nsmallest(10).keys()
    print('\n')
    print(wv.get_word_at_indices(list(res)))


if __name__=="__main__":
    k_nearest(sys.argv[1], sys.argv[2], sys.argv[3])
