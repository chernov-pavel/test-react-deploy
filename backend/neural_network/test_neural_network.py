import numpy as np

from neural_network.configuration import SEQUENCE_LENGTH

np.random.seed(42)
import tensorflow as tf
tf.set_random_seed(42)
from keras.models import load_model
import matplotlib.pyplot as plt
import pickle
import heapq
import seaborn as sns
from pylab import rcParams


def sample(preds, top_n=3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)

    return heapq.nlargest(top_n, range(len(preds)), preds.take)


def prepare_input(text):
    x = np.zeros((1, SEQUENCE_LENGTH, len(chars)))
    for t, char in enumerate(text):
        x[0, t, char_indices[char]] = 1.

    return x


def predict_completion(text):
    original_text = text
    generated = text
    completion = ''
    while True:
        x = prepare_input(text)
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, top_n=1)[0]
        next_char = indices_char[next_index]
        text = text[1:] + next_char
        completion += next_char

        if len(original_text + completion) + 2 > len(original_text) and next_char == ' ':
            return completion


def predict_completions(text, n=3):
    x = prepare_input(text)
    preds = model.predict(x, verbose=0)[0]
    next_indices = sample(preds, n)
    return [indices_char[idx] + predict_completion(text[1:] + indices_char[idx]) for idx in next_indices]


if __name__ == '__main__':
    sns.set(style='whitegrid', palette='muted', font_scale=1.5)

    rcParams['figure.figsize'] = 12,5

    path = 'data\\dataset.txt'
    text = open(path).read()
    print('corpus length:', len(text))

    chars = sorted(list(set(text)))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    print(f'unique chars: {len(chars)}')

    model = load_model('keras_model.h5')
    history = pickle.load(open("history.p", "rb"))

    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left');
    plt.show()

    quotes = [
        "Однажды весною в час небывало жаркого заката в Москве на Патриарших прудах.",
        "Есть ли вероятность того что дальше будет разгадано данное слово.",
        "Кто виноват и как с этим быть ответит наш эксперт по вопросам молодежной политики"
    ]

    for q in quotes:
        seq = q[:SEQUENCE_LENGTH].lower()
        print(seq)
        print(predict_completions(seq, 5))
        print()

    print('Done')