import numpy as np
from scipy.spatial import distance

from basescript import BaseScript as BS
from diskarray import DiskArray as DA

class EasyTripletVerification(BS):
    DIM = 300
    ALPHA = np.linspace(10000, 100000, 100)

    def __init__(self):
        super(EasyTripletVerification, self).__init__()
        self.train_d = DA(self.args.train_f, dtype=self._get_dtype())

    def _get_dtype(self):
        return [
                    ('anchor', np.float32, self.DIM),
                    ('positive', np.float32, self.DIM),
                    ('negative', np.float32, self.DIM),
            ]

    def run(self):
        for alpha_val in self.ALPHA:
            total_violations = self.check_violation(alpha_val)
            if total_violations > 0:

                print(
                        'For alpha:', alpha_val,
                        'total', total_violations,
                        'violations found, out of', len(self.train_d[:]), 'samples',
                    )

    def check_violation(self, alpha):
        for i in range(len(self.train_d[:])):
            constraint_violated = 0

            anchor = self.train_d['anchor'][i].reshape(1, self.DIM)
            positive = self.train_d['positive'][i].reshape(1, self.DIM)
            negative = self.train_d['negative'][i].reshape(1, self.DIM)

            dist_anch_pos = np.sum(np.square(anchor - positive), axis=1)
            dist_anch_neg = np.sum(np.square(anchor - positive), axis=1)

            if dist_anch_pos + alpha > dist_anch_neg:
                constraint_violated += 1

            return constraint_violated

    def define_args(self, parser):
        parser.add_argument('train_f', help='training diskarray')


if __name__ == '__main__':
    EasyTripletVerification().start()
