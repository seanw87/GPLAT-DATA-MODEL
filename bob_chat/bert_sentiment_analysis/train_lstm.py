'''Trains an LSTM model on the IMDB sentiment classification task.

The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.

# Notes

- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.

- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tokenization import FullTokenizer

path = "./"
pd_all = pd.read_csv(os.path.join(path, "weibo_senti_100k.csv"))

tokenizer = FullTokenizer("vocab.txt")

pd_all = shuffle(pd_all)

x_data, y_data = pd_all.review.values, pd_all.label.values

x_data = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text)) for text in x_data]

x_train, x_test, y_train, y_test = train_test_split(np.array(x_data), y_data, test_size=0.2)

max_features = 21128
# cut texts after this number of words (among top max_features most common words)
maxlen = 128
batch_size = 32

print('Loading data...')
#(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                                    metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
                  batch_size=batch_size,
                            epochs=15,
                                      validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                                    batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
