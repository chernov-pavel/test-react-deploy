import numpy as np

from backend.settings import BASE_DIR
from neural_network.configuration import SEQUENCE_LENGTH

np.random.seed(42)
import tensorflow as tf

tf.set_random_seed(42)
from keras.models import load_model
import pickle
import heapq


class NeuralNetwork:
    instance = None

    def __init__(self):
        print(f'Init NeuralNetwork')
        path = BASE_DIR + '\\neural_network\\data\\dataset.txt'
        text = open(path).read()
        self.chars = sorted(list(set(text)))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        self.model = load_model(BASE_DIR + '\\neural_network\\keras_model.h5')
        self.history = pickle.load(open(BASE_DIR + "\\neural_network\\history.p", "rb"))


    def get_instance():
        if NeuralNetwork.instance is None:
            NeuralNetwork.instance = NeuralNetwork()
        return NeuralNetwork.instance

    def __sample(self, preds, top_n=3):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds)
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)

        return heapq.nlargest(top_n, range(len(preds)), preds.take)

    def __prepare_input(self, text):
        x = np.zeros((1, SEQUENCE_LENGTH, len(self.chars)))
        for t, char in enumerate(text):
            x[0, t, self.char_indices[char]] = 1.

        return x

    def __predict_completion(self, text):
        original_text = text
        generated = text
        completion = ''
        while True:
            x = self.__prepare_input(text)
            preds = self.model.predict(x, verbose=0)[0]
            next_index = self.__sample(preds, top_n=1)[0]
            next_char = self.indices_char[next_index]
            text = text[1:] + next_char
            completion += next_char

            if len(original_text + completion) + 2 > len(original_text) and next_char == ' ':
                return completion

    def predict_completions(self, text, n=3):
        x = self.__prepare_input(text)
        preds = self.model.predict(x, verbose=0)[0]
        next_indices = self.__sample(preds, n)
        return [self.indices_char[idx] + self.__predict_completion(text[1:] + self.indices_char[idx]) for idx in next_indices]
