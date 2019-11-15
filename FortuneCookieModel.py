#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/WomenInDataScience-Seattle/FortuneCookie/blob/master/FortuneCookieModel.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


from io import StringIO

import numpy as np
import pandas as pd
import requests
url='https://raw.githubusercontent.com/WomenInDataScience-Seattle/Machine_Learning_Projects/master/FortuneCookie/training_data/data.csv'
s=requests.get(url).text

c=pd.read_csv(StringIO(s))


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.spatial.distance import cdist


# In[ ]:



# from tf.keras.models import Sequential  # This does not work!
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding, LSTM, Dropout
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import keras.utils as ku 


# In[ ]:


# random-word used to generate the first word in the sequence
get_ipython().system('pip install random-word')
from random_word import RandomWords


# In[ ]:


c.head(5)


# In[ ]:


fortune_data = c['Fortune Cookie Quotes']


# In[ ]:


fortune_data.head(5)


# In[ ]:


fortune_data[1]


# In[ ]:


fortune_data[36]


# In[ ]:


cleaned_df = fortune_data.str.lower()
cleaned_df2 = cleaned_df.str.strip()


# In[ ]:


dropped = cleaned_df2.dropna()


# In[ ]:


dropped.tail(5)


# In[ ]:


cleaned_fortunes = dropped


# In[ ]:


cleaned_fortunes.head(5)


# In[ ]:


cleaned_fortunes[3]


# In[ ]:


cleaned_fortunes[0]


# In[ ]:


corpus = cleaned_fortunes


# In[ ]:


tokenizer = Tokenizer()

def get_sequence_of_tokens(corpus):
    ## tokenization
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    
    ## convert data to sequence of tokens 
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    return input_sequences, total_words

inp_sequences, total_words = get_sequence_of_tokens(corpus)
inp_sequences[:10]


# In[ ]:


def generate_padded_sequences(input_sequences):
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    
    predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
    label = ku.to_categorical(label, num_classes=total_words)
    return predictors, label, max_sequence_len

predictors, label, max_sequence_len = generate_padded_sequences(inp_sequences)


# In[ ]:


predictors[60]


# In[ ]:


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

model = create_model(max_sequence_len, total_words)
model.summary()


# In[ ]:


model.fit(predictors, label, epochs=100, verbose=5)


# In[ ]:


# the original generate text function from https://www.kaggle.com/shivamb/beginners-guide-to-text-generation-using-lstms

def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        
        output_word = ""
        for word,index in tokenizer.word_index.items():
            print(predicted)
            print(np.sum(predicted))
            if index == predicted:
                output_word = word
                break
        seed_text += " "+output_word
    return seed_text.title()
  


# In[ ]:


# tweaked generate text function that uses np.random.choice to sample of the probaility distribution of the predicted word

def generate_text_prob(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict_proba(token_list, verbose=0)
        random = np.random.choice(predicted.shape[1],1, p=predicted[0])
        
        output_word = ""
        for word,index in tokenizer.word_index.items():
            if index == random:
                output_word = word
                break
        seed_text += " "+output_word
    return seed_text.title()
  


# In[ ]:


token_list = tokenizer.texts_to_sequences('you')[0]
token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
predicted = model.predict_proba(token_list, verbose=0)
random = np.random.choice(predicted.shape[1],1, p=predicted[0])

print(random)
predicted[0].shape


# In[ ]:


r = RandomWords()
random_word = 'Dreams'
text = generate_text_prob(random_word, 7, model, max_sequence_len)
print(text)


# What we did today: 
# - we changed to gru 
# - we increased the word embedding length
# - we increased the dropout
# - we changed the activation from tanh to relu
# - we randomly sampled our probaility distribution of word predictions
# 
# Next time:
# - Use a pre-trained word embedding applied to our corpus
# - get more data
# - try training
# 
