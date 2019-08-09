from sys import argv as A
import json
import random

import numpy as np

train_set = open(A[1], 'w+')
test_set = open(A[2], 'w+')

X = lambda x: round(random.uniform(-1, 1), 4)

rand_total =  [[X(i) for i in range(200)] for j in range(1500000)]

rand_test = []
rand_train = []
for i in range(1, len(rand_total)-5, 2):

	rand_train.append(rand_total[i+1])
	try:
		rand_test.append(rand_total[i+2])
	except:
		rand_test.append(rand_total[i])


train_set.write(json.dumps(rand_train))
test_set.write(json.dumps(rand_test))

train_set.close()
test_set.close()

