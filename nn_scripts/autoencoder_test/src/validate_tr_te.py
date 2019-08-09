from sys import argv as A
import json

import numpy as np

train_f = open(A[1])
test_f = open(A[2])

train_d = json.loads(train_f.read())

tr_idx = {}
for k, v in enumerate(train_d):
    k = str(k)
    v = str(v)
    tr_idx[v] = k


dup_samples = []

for line in test_f:
    samples = json.loads(line)
    for sample in samples:
        sample = str(sample)
        try:
            tr_idx[sample]
            dup_samples.append('sample')
            import pdb; pdb.set_trace()
        except KeyError:
            print('No Duplicate Found Yet')


