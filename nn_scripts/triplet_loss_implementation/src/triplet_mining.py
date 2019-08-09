import numpy as np
from scipy.spatial.distance import Euclidean

from wordvecspace import WordVecSpaceMem
from basescript import BaseScript as BS
from diskarray import DiskArray as DA
from diskdict import DiskDict as DD

class DistanceMatrix(object):
    def __init__(self, cluster_union, vspace):
        self.cluster = cluster_union
        self.vspace = vspace

    def compute_distances(self):
        pass

class MineTriplet(BS):
    def __init__(self):
        super(MineTriplet, self).__init__()

        self.inp_cluster_f = DD(self.args.manual_cluster_f)
        self.vspace = WordVecSpaceMem(self.args.wvspace_f)
        self.out_train_d = DA(self.args.hard_triplet_batch, shape=(0,), dtype=self._get_dtype())

    def _get_dtype(self):
        return [
                    ('vec1', np.float32, 300),
                    ('vec2', np.float32, 300),
                    ('label', np.int, 1),
            ]

    def run(self):
        batched_clusters = self.get_batched_clusters(self.args.batch_size)
        clusters_union = self.get_cluster_union(batched_clusters)
        distance_matrix = DistanceMatrix(clusters_union, self.vspace)

    def get_batched_clusters(self, batch_size):
        cluster_iter = 0
        positives = []

        for values in self.inp_cluster_f.values():
            if cluster_iter < batch_size:
                positives.append(values['positive'])
                cluster_iter += 1

        return positives

    def get_cluster_union(self, batched_clusters):
        clusters_union = set().union(*batched_clusters)

        return clusters_union

    def define_args(self, parser):
        parser.add_argument('manual_cluster_f', help='manual cluster file')
        parser.add_argument('wvspace_f', help='vector space file')
        parser.add_argument('--batch_size', default=5, type=int,
                                help='size to produce triplets')
        parser.add_argument('hard_triplet_batch', help='batch of training triplets')

if __name__ == '__main__':
    MineTriplet().start()

