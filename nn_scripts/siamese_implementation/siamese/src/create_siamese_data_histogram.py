import csv

import numpy as np
import random as rn

from scipy.spatial import distance as D


import matplotlib
import matplotlib.pyplot as plt

from diskarray import DiskArray
from basescript import BaseScript

class Generate_Histogram(BaseScript):

    def __init__(self):
        super(Generate_Histogram, self).__init__()
        self.d = DiskArray(self.args.array, dtype=self._get_dtype())
        self.psame = []
        self.pnsame = []
        csv_f = open(self.args.csv_f, 'w')
        self.csv_f = csv.writer(csv_f)

    def _get_dtype(self):
        d_type = [
            ('vec1', np.float32, 300),
            ('vec2', np.float32, 300),
            ('label', np.int, 1),
        ]

        return d_type

    def run(self):
        self.csv_f.writerow(['index', 'ps_dist', 'pns_dist'])

        for i, row in enumerate(self.d):
            vec1 = row['vec1']
            vec2 = row['vec2']
            label = row['label']

            r_vec1 = vec1.reshape(1, len(a))
            r_vec2 = vec2.reshape(1, len(p))

            if self.args.metric == 'euclidean':
                p_d = D.euclidean(r_vec1, r_vec2)
            else:
                p_d = D.cosine(r_vec1, r_vec2)

            self.csv_f.writerow([i, p_d, label])

            if label == 0:
                self.psame.append(p_d)
            else:
                self.pnsame.append(p_d)

        self.plot_histogram()

    def plot_histogram(self):

        plt.hist(self.psame, bins=50, alpha=0.5, label='Anchor-Positive', color='black')
        plt.hist(self.pnsame, bins=50, alpha=0.5, label='Anchor-Negative', color='thistle')
        plt.legend(loc='upper right')
        plt.savefig(self.args.outf)

    def define_args(self, parser):
        parser.add_argument('array', help="training or testing array file")
        parser.add_argument('outf', help="output image name")
        parser.add_argument('csv_f', help="out csv file")
        parser.add_argument('metric', help="metric formula")

if __name__ == '__main__':
    Generate_Histogram().start()
