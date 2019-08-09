import sys
import json

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np
from scipy.spatial import distance

from wordvecspace import WordVecSpaceMem
from diskarray import DiskArray

def k_nearest(wvspace, disk_f, words, metric, image_name):
    f = open(words)
    wv = WordVecSpaceMem(wvspace)
    da = DiskArray(disk_f, dtype=[('vec', np.float32, 300)])

    vecs = da['vec']

    psame = []
    pnsame = []
    for line in f:
        words = json.loads(line.strip())
        index1 = wv.get_word_index(words[0])
        index2 = wv.get_word_index(words[1])

        if 'clinicaltrials' in words[0] or 'clinicaltrials' in words[1]:
            continue

        vec1 = vecs[index1].reshape(1, 300)
        vec2 = vecs[index2].reshape(1, 300)

        if metric == 'cosine':
           vspace_dist = wv.get_distance(words[0], words[1])
           tvspace_dist = distance.cosine(vec1, vec2)
        else:
            vspace_dist = wv.get_distance(words[0], words[1], metric='euclidean')
            tvspace_dist = distance.euclidean(vec1, vec2)

        if words[2] == 0:
            psame.append(tvspace_dist)
        else:
            pnsame.append(tvspace_dist)

    dm = (np.std(psame) + np.std(pnsame)) / 2
    nm = abs(np.mean(psame) - np.mean(pnsame))

    d = nm / dm
    print('the cohens D distance is', d)

    plt.hist(psame, bins=50, alpha=0.5, label='same points')
    plt.hist(pnsame, bins=50, alpha=0.5, label='not same points')
    plt.legend(loc='upper right')
    plt.savefig(image_name)


if __name__=="__main__":
    k_nearest(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
