import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, SimpleRNN, Input, LSTM
from keras.utils import to_categorical

from basescript import BaseScript

class Rnn(BaseScript):

    def __init__(self):
        super(Rnn, self).__init__()
        self.vocabs = None
        self.units = 100

    def char_to_int(self, chars):
        return dict((c, i+1) for i, c in enumerate(chars))

    def int_to_char(self, chars):
        return dict((i+1, c) for i, c in enumerate(chars))

    def get_unique_chars(self, text):
        chars = sorted(list(set(text)))
        return chars

    def get_sequences(self, char_to_int):

        training_data = []
        data = open(self.args.trainf)
        for word in data:
            word = '\t' + word.strip() + '\n'
            seq_data = np.zeros(self.units)
            for i, char in enumerate(word):
                seq_data[i] = char_to_int[char]
            seq_data.tolist()
            training_data.append(seq_data)

        training_data = np.array(training_data)

        target_data = []
        for x in training_data:
            tar_seq = []
            for i in x[1:]:
                tar_seq.append(i)
            tar_seq.append(0)
            target_data.append(tar_seq)

        target_data = np.array(target_data)

        return training_data, target_data

    def _get_one_hot_vectors(self, sequence):
        neasted_data = []
        for num in sequence:
            data = np.zeros(self.vocabs)
            if int(num) >= 1:
                data[int(num)] = 1
            neasted_data.append(data)

        return neasted_data

    def get_one_hot_vectors(self, training_data, target_data):
        train_vectors = to_categorical(training_data, self.vocabs)
        training_data = np.array(train_vectors)

        target_vectors = to_categorical(target_data, self.vocabs)
        target_data = np.array(target_vectors)

        return training_data, target_data

    def create_model(self, data):
        model = Sequential()
        model.add(SimpleRNN(256, input_shape=(data.shape[1], data.shape[2])))
        model.add(Dense(self.vocabs, activation='softmax'))
        return model

    def compile_model(self, model):
        model.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['mae'])
        return model

    def train_model(self, model, trainx, testx):
        model.fit(trainx, testx, epochs=self.args.epochs, batch_size=32)

        return model

    def run(self):
        train_text = open(self.args.trainf).read().strip()

        train_text = train_text + '\t' + '\n'
        unique_chars = self.get_unique_chars(train_text)
        self.vocabs = len(unique_chars) + 1

        chars_to_int = self.char_to_int(unique_chars)
        training_data, target_data = self.get_sequences(chars_to_int)
        training_data, target_data = self.get_one_hot_vectors(training_data, target_data)
        model = self.create_model(training_data)
        model = self.compile_model(model)
        model = self.train_model(model, training_data, target_data)

    def define_args(self, parser):
        parser.add_argument('trainf', help='input file')
        parser.add_argument('epochs', type=int, help='num of epochs')

if __name__ == '__main__':
    Rnn().start()
