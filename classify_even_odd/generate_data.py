def generate_data():
    for i in range(1000):
        if i % 2 == 0:
            print([i, 0, 0])
        else:
            print([i, 1, 1])


generate_data()
