import sys
import json

import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dense

def create_model(input_dms):
   model = Sequential()
   model.add(Dense(2, input_shape=(input_dms,), activation="softplus"))
   model.add(Dense(1, activation="softplus"))

   return model

def get_training_data(f):
    train_d, labels = [], []
    for line in f:
        line = json.loads(line)
        train_d.append([line[0], line[1]])
        labels.append(line[2])

    return np.array(train_d), np.array(labels)

def compile_model(model):
    model.compile(loss="mse", optimizer="adam")

    return model

def train_model(model, train_d, labels):
    model.fit(train_d, labels, epochs=10000)

    return model

def predict(model, t):
    val = model.predict(t.reshape(1, len(t)))

    print(val)

def start():
    training_f = open(sys.argv[1])
    train_d, labels = get_training_data(training_f)
    model = create_model(len(train_d[0]))
    model = compile_model(model)
    model = train_model(model, train_d, labels)

    predict(model, np.array([666, 333]))

if __name__ == "__main__":
    start()
