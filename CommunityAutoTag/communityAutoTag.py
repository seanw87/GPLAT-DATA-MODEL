from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import jieba
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow import keras
from etl import etl

layers = keras.layers
Sequential = keras.models.Sequential
Dense = keras.layers.Dense
Embedding = keras.layers.Embedding
GlobalMaxPool1D = keras.layers.GlobalMaxPool1D
Dropout = keras.layers.Dropout
Adam = keras.optimizers.Adam
Conv1D = keras.layers.Conv1D
Activation = keras.layers.Activation

ReduceLROnPlateau = keras.callbacks.ReduceLROnPlateau
EarlyStopping = keras.callbacks.EarlyStopping
ModelCheckpoint = keras.callbacks.ModelCheckpoint

print("You have TensorFlow version", tf.__version__)

prediction_simple_model_file = "model/community_auto_tag_simple.h5"
prediction_cnn1d_model_file = "model/community_auto_tag_cnn1d.h5"
dataset_file = "dataset/posts/posts_labeled.csv"
dataset_resampled_file = "dataset/posts/posts_labeled_resampled.csv"
dataset_labels_file = "dataset/posts/labels.csv"

# etl()

data = pd.read_csv(dataset_resampled_file)

# 1. content seq
content_cut = data['content_cut'].astype(str)

vocab_size = 12000
tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size, char_level=False)
tokenizer.fit_on_texts(content_cut)

embed = tokenizer.texts_to_sequences(content_cut)
max_seq_length = 170
x = keras.preprocessing.sequence.pad_sequences(embed, maxlen=max_seq_length, padding="post")

# 2. multi labels
multi_labels = []
def gen_multi_labels(labels):
    labels_arr = labels.split(",")
    label_tuple = ()
    for label in labels_arr:
        label_tuple = label_tuple + (label, )
    multi_labels.append(label_tuple)
data.apply(lambda row: gen_multi_labels(row["labelID"]), axis=1)

multilabel_binarizer = MultiLabelBinarizer()
y = multilabel_binarizer.fit_transform(multi_labels)
labels = multilabel_binarizer.classes_
print(labels)

# store labels
classes_dataframe = pd.DataFrame(labels, columns=["class"])
classes_dataframe.to_csv(dataset_labels_file)

num_classes = len(labels)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=9000)


# model construction

# simple model
model = Sequential()
model.add(Embedding(vocab_size, 20, input_length=max_seq_length))
model.add(Dropout(0.15))
model.add(GlobalMaxPool1D())
model.add(Dense(num_classes, activation='sigmoid'))

# model.compile(optimizer=Adam(0.015), loss='binary_crossentropy', metrics=['categorical_accuracy'])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['categorical_accuracy'])
callbacks = [
    ReduceLROnPlateau(),
    EarlyStopping(patience=4),
    ModelCheckpoint(filepath=prediction_simple_model_file, save_best_only=True)
]

model.fit(x_train, y_train,
                #class_weight=class_weight,
                epochs=20,
                batch_size=32,
                validation_split=0.1,
                callbacks=callbacks)
simple_model = keras.models.load_model(prediction_simple_model_file)
metrics = simple_model.evaluate(x_test, y_test)
print("{}: {}".format(simple_model.metrics_names[0], metrics[0]))
print("{}: {}".format(simple_model.metrics_names[1], metrics[1]))


# 1dCNN
filter_length = 300

model = Sequential()
model.add(Embedding(vocab_size, 20, input_length=max_seq_length))
model.add(Dropout(0.1))
model.add(Conv1D(filter_length, 3, padding='valid', activation='relu', strides=1))
model.add(GlobalMaxPool1D())
model.add(Dense(num_classes))
model.add(Activation('sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['categorical_accuracy'])
model.summary()

callbacks = [
    ReduceLROnPlateau(),
    EarlyStopping(patience=4),
    ModelCheckpoint(filepath=prediction_cnn1d_model_file, save_best_only=True)
]


# evaluate
model.fit(x_train, y_train,
                # class_weight=class_weight,
                epochs=20,
                batch_size=32,
                validation_split=0.1,
                callbacks=callbacks)

cnn_model = keras.models.load_model(prediction_cnn1d_model_file)
metrics = cnn_model.evaluate(x_test, y_test)
print("{}: {}".format(cnn_model .metrics_names[0], metrics[0]))
print("{}: {}".format(cnn_model .metrics_names[1], metrics[1]))