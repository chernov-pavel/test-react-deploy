import numpy as np

from neural_network.configuration import *

np.random.seed(42)
import tensorflow as tf
tf.set_random_seed(42)
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.core import Dense, Activation
from keras.optimizers import RMSprop
import pickle


chars = []
char_indices = {}
indices_char = {}

if __name__ == '__main__':
    path = 'data\\dataset.txt'
    text = open(path).read()
    print('corpus length:', len(text))

    chars = sorted(list(set(text)))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    print(f'unique chars: {len(chars)}')

    sentences = []
    next_chars = []
    for i in range(0, len(text) - SEQUENCE_LENGTH, TRAIN_STEP):
        sentences.append(text[i: i + SEQUENCE_LENGTH])
        next_chars.append(text[i + SEQUENCE_LENGTH])

    print(f'num training examples: {len(sentences)}')

    X = np.zeros((len(sentences), SEQUENCE_LENGTH, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    model = Sequential()
    model.add(LSTM(128, input_shape=(SEQUENCE_LENGTH, len(chars))))
    model.add(Dense(len(chars)))
    model.add(Activation(ACTIVATION_FUNCTION))

    optimizer = RMSprop(lr=LEARNING_RATE)
    model.compile(loss=LOSS, optimizer=optimizer, metrics=['accuracy'])
    history = model.fit(X, y, validation_split=VALIDATION_SPLIT, batch_size=BATCH_SIZE, epochs=EPOCHS_COUNT, shuffle=True).history

    model.save('keras_model.h5')
    pickle.dump(history, open("history.p", "wb"))