#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/WomenInDataScience-Seattle/FortuneCookie/blob/master/FortuneCookieModel.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import Dense, GRU, Embedding, LSTM, Dropout
from tensorflow.python.keras.models import Sequential
from scipy.spatial.distance import cdist
import tensorflow as tf
from io import StringIO

import numpy as np
import pandas as pd
import requests
import os

from library.common import get_sequence_of_tokens, get_word_index, generate_embedding_matrix, generate_padded_sequences, create_model_glove_embedding, generate_text
# Glove embedding layer constants
BASE_DIR = './training_data/'
GLOVE_DIR = os.path.join(BASE_DIR, '')
TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup')


# 41 is the max length of the sentence - 1
MAX_SEQUENCE_LENGTH = 41 # used to be 1000
MAX_NUM_WORDS = 20000

# number of the dimensions for each word
EMBEDDING_DIM = 100 #used to be 100
VALIDATION_SPLIT = 0.2

def generate_embeddings_index(GLOVE_DIR):
    embeddings_index = {}
    with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs
    return embeddings_index

# When we go to add the embedding layer, use logic similar to the way this package does a pull of
# it's embedding. https://github.com/minimaxir/gpt-2-simple/blob/master/README.md#usage
# Search for "if not os.path.isfile(file_name):"

# TODO: Move this to `init.py`

# update the url for the csv
fortune_cookie_csv_url = 'https://raw.githubusercontent.com/WomenInDataScience-Seattle/FortuneCookie/master/training_data/data.csv'
fortune_cookie_string = requests.get(fortune_cookie_csv_url).text
fortune_cookie_df = pd.read_csv(StringIO(fortune_cookie_string))
# Extract the column of csv data that we want.
corpus = fortune_cookie_df['Fortune Cookie Quotes']


inp_sequences, total_words = get_sequence_of_tokens(corpus)
word_index = get_word_index(corpus)
embeddings_index = generate_embeddings_index(GLOVE_DIR)
embedding_layer = generate_embedding_matrix(total_words, word_index, MAX_NUM_WORDS, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, embeddings_index)

predictors, label, max_sequence_len = generate_padded_sequences(
    inp_sequences, total_words)
model = create_model_glove_embedding(max_sequence_len, total_words, embedding_layer)
model.fit(predictors, label, epochs=1, verbose=5)


random_word = 'Dreams'
text = generate_text(random_word, 7, model, max_sequence_len)
print(text)
