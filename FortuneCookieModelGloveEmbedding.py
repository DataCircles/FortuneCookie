import os
from library.common import get_fortune_cookie_corpus, get_sequence_of_tokens, generate_embeddings_index, get_word_index, generate_embedding_matrix, generate_padded_sequences, create_model_glove_embedding, generate_text

# Glove embedding layer constants
BASE_DIR = './training_data/'
GLOVE_DIR = os.path.join(BASE_DIR, '')
TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup')


# 41 is the max length of the sentence - 1
MAX_SEQUENCE_LENGTH = 41 # used to be 1000
MAX_NUM_WORDS = 20000

# number of the dimensions for each word
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2


# pre-processing for the model
corpus = get_fortune_cookie_corpus()
input_sequences, total_words = get_sequence_of_tokens(corpus)
word_index = get_word_index(corpus)
embeddings_index = generate_embeddings_index(GLOVE_DIR)
embedding_layer = generate_embedding_matrix(total_words, word_index, MAX_NUM_WORDS, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, embeddings_index)
predictors, label, max_sequence_len = generate_padded_sequences(
    input_sequences, total_words)

# create model
model = create_model_glove_embedding(max_sequence_len, total_words, embedding_layer)

# fit model
model.fit(predictors, label, epochs=1, verbose=5)


# generate a fortune
random_word = 'Dreams'
text = generate_text(random_word, 7, model, max_sequence_len)
print(text)
