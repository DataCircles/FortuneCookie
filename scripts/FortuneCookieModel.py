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

from library.common import get_sequence_of_tokens, generate_padded_sequences, create_model, generate_text

# When we go to add the embedding layer, use logic similar to the way this package does a pull of
# it's embedding. https://github.com/minimaxir/gpt-2-simple/blob/master/README.md#usage
# Search for "if not os.path.isfile(file_name):"

# TODO: Move this to `init.py`
fortune_cookie_csv_url = 'https://raw.githubusercontent.com/WomenInDataScience-Seattle/Machine_Learning_Projects/master/FortuneCookie/training_data/data.csv'
fortune_cookie_string = requests.get(fortune_cookie_csv_url).text
fortune_cookie_df = pd.read_csv(StringIO(fortune_cookie_string))
# Extract the column of csv data that we want.
corpus = fortune_cookie_df['Fortune Cookie Quotes']


inp_sequences, total_words = get_sequence_of_tokens(corpus)
predictors, label, max_sequence_len = generate_padded_sequences(
    inp_sequences, total_words)
model = create_model(max_sequence_len, total_words)
model.fit(predictors, label, epochs=1, verbose=5)


random_word = 'Dreams'
text = generate_text(random_word, 7, model, max_sequence_len)
print(text)
