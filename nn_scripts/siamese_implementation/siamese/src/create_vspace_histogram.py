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
            ('anchor', np.float32, 300),
            ('positive', np.float32, 300),
            ('negative', np.float32, 300),
        ]

        return d_type

    def run(self):
        self.csv_f.writerow(['index', 'ps_dist', 'pns_dist'])

        for i, row in enumerate(self.d):
            a = row['anchor']
            p = row['positive']
            n = row['negative']

            r_a = a.reshape(1, len(a))
            r_p = p.reshape(1, len(p))
            r_n = n.reshape(1, len(n))

            if self.args.metric == 'euclidean':
                ps_d = D.euclidean(r_a, r_p)
                pns_d = D.euclidean(r_a, r_n)
            else:
                ps_d = D.cosine(r_a, r_p)
                pns_d = D.cosine(r_a, r_n)

            self.csv_f.writerow([i, ps_d, pns_d])

            self.psame.append(ps_d)
            self.pnsame.append(pns_d)

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
