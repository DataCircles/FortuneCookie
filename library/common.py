import keras.utils as ku
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import Dense, GRU, Embedding, LSTM, Dropout
from tensorflow.python.keras.models import Sequential
from scipy.spatial.distance import cdist
import tensorflow as tf
import requests
import os
import numpy as np
import pandas as pd
from random import choice
from io import StringIO


source_csv = 'https://raw.githubusercontent.com/WomenInDataScience-Seattle/FortuneCookie/master/training_data/data.csv'

def get_fortune_cookie_corpus(fortune_cookie_csv_url = source_csv):
    fortune_cookie_string = requests.get(fortune_cookie_csv_url).text
    fortune_cookie_df = pd.read_csv(StringIO(fortune_cookie_string))
    return fortune_cookie_df['Fortune Cookie Quotes']

 
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


def generate_embeddings_index(GLOVE_DIR):
    embeddings_index = {}
    with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs
    return embeddings_index


# return word index
def get_word_index(corpus):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    return tokenizer.word_index


# prepare embedding matrix
def generate_embedding_matrix(total_words, word_index, MAX_NUM_WORDS, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, embeddings_index):
    num_words = min(MAX_NUM_WORDS, total_words)
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(total_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    return embedding_layer


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


def create_model_glove_embedding(max_sequence_len, total_words,embedding_layer):
    input_len = max_sequence_len - 1
    model = Sequential()

    # Add input embedding Layer
    model.add(embedding_layer)

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