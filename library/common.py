import keras.utils as ku
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import Dense, GRU, Embedding, LSTM, Dropout
from tensorflow.python.keras.models import Sequential
from scipy.spatial.distance import cdist
import tensorflow as tf

import numpy as np
import pandas as pd
from random import choice

tokenizer = Tokenizer()
def get_sequence_of_tokens(corpus):
    # tokenization
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    # convert data to sequence of tokens
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    return input_sequences, total_words

def generate_padded_sequences(input_sequences, total_words):
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(
        input_sequences, maxlen=max_sequence_len, padding='pre'))

    predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
    label = ku.to_categorical(label, num_classes=total_words)
    return predictors, label, max_sequence_len

def create_model(max_sequence_len, total_words):
    input_len = max_sequence_len - 1
    model = Sequential()

    # Add Input Embedding Layer
    model.add(Embedding(total_words, 50, input_length=input_len))

    # Add Hidden Layer 1 - LSTM Layer
    model.add(GRU(100, activation='relu'))
    model.add(Dropout(0.2))

    # Add Output Layer
    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model

# Drew inspiration from https://www.kaggle.com/shivamb/beginners-guide-to-text-generation-using-lstms
# Tweaked generate text function that uses np.random.choice to sample of the probaility distribution of the predicted word

def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences(
            [token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict_proba(token_list, verbose=0)
        random = np.random.choice(predicted.shape[1], 1, p=predicted[0])

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == random:
                output_word = word
                break
        seed_text += " "+output_word
    return seed_text.title()


def _random_prefix(sentences):
    """
    prefix random generator
    input: list of input sentences
    output: random word
    """

    words = _word_dict(sentences)
    return choice(words)

def _word_dict(sentences):
    """
    input: list of input sentences
    output: unique list of the corpus
    #to-do: strip out punctuation
    """
    result = []
    for i in range(len(sentences)):
        sen_list = set(sentences[i].split().lower())
        for word in sen_list:
            if word not in result:
                result.append(word)
            else:
                pass
    return result