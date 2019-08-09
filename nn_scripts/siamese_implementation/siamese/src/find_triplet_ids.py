import sys

indexes_f = open(sys.argv[1])
triples = open(sys.argv[2])

indexes = []
for index in indexes_f:
    index = index.strip()
    indexes.append(index)

for i, line in enumerate(triples):
    if str(i) in indexes:
        line = line.strip()
        line = line.split(',')
        print(line[0], line[1], line[2])

