import requests, numpy as np, string, pandas as pd

#important variables
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 40000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2

#fit the tokenizer
from keras.preprocessing.text import Tokenizer

all_text = []
for chapter in np.arange(1, 19):  # 18 chapters in ulysses

    # get the book
    if chapter < 18:
        url = 'http://faculty.georgetown.edu/jod/ulysses/ulys%d.txt' % chapter
    else:
        url = 'http://faculty.georgetown.edu/jod/ulysses/ulys18dr.txt'
    response = requests.get(url)

    all_text.append(response.text)

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)

tokenizer.fit_on_texts(all_text)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


#turn text into series of sentences
from nltk import wordpunct_tokenize
from nltk import sent_tokenize

punct = string.punctuation

sequences, ch_label = [], []
for chapter in np.arange(1, 19):  # 18 chapters in ulysses

    # get the book
    if chapter < 18:
        url = 'http://faculty.georgetown.edu/jod/ulysses/ulys%d.txt' % chapter
    else:
        url = 'http://faculty.georgetown.edu/jod/ulysses/ulys18dr.txt'
    response = requests.get(url)

    for sent in sent_tokenize(response.text):
        sent_words = []
        for token in wordpunct_tokenize(sent):

            # Apply preprocessing to the token
            token = token.lower()
            token = token.strip()
            token = token.strip('_')
            token = token.strip('*')

            # remove numbers
            if token.isdigit():
                continue

            # If punctuation, ignore token and continue
            if all(char in punct for char in token):
                continue

            if token:
                sent_words.append(token)

        if len(sent_words) > 0:
            sequences.append(tokenizer.texts_to_sequences(sent_words))
            ch_label.append(chapter)

#change data to shape and format for model
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

num_chapters = len(np.unique(ch_label))

hold_data = np.array([pad_sequences(sent) for sent in sequences])

data, chapter_label = [],[]
for chapter,sent in zip(ch_label,hold_data):
    if np.shape(sent.T)[0] == 1:
        hold = np.zeros((1000,))
        if len(sent) > 1000: sent = sent[:1000]
        hold[-len(sent):] = np.squeeze(sent)
        data.append(np.array(hold))
        hold = np.zeros((num_chapters,))
        hold[chapter-1] = 1
        chapter_label.append(hold)
    else:
        continue

data = np.array(data)
chapter_label = np.array(chapter_label)
#chapter_label = to_categorical(np.asarray(ch_label))

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', chapter_label.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
chapter_label = chapter_label[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = chapter_label[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = chapter_label[-nb_validation_samples:]


#get embedding
import os

BASE_DIR = '.'
GLOVE_DIR = BASE_DIR + '/glove.6B/'
#I would like to use one of the larger ones given the huge vocab of this book, but i don't have the ram

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

#create embedding matrix
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


#create neural net embedding
from keras.layers import Embedding

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)


#create and train the model
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Model


sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)  # global max pooling
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dense(128, activation='relu')(x)
preds = Dense(num_chapters, activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

# happy learning!
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=5, batch_size=128)