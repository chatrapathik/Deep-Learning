import random

def generate_data():
    l1 = [random.randrange(100, 300) for i in range(100)]
    l2 = [random.randrange(400, 600) for i in range(100)]
    for i in range(len(l1)):
        v1 = l1[i]
        v2 = l2[i]
        val = sum([v1, v2])
        print([v1, v2, val])

generate_data()
