from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import jieba
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
layers = keras.layers

print("You have TensorFlow version", tf.__version__)

#初始化配置
prediction_model_file = "model/chat_filtering.h5"
dataset_file = "dataset/chat-filtering/chat_labeled.csv"
dataset_resampled_file = "dataset/chat-filtering/chat_labeled_resampled.csv"
jieba_dict_file = "dataset/jieba-dict/dict.txt"

jieba.load_userdict(jieba_dict_file)

# resample the original data set or get the resampled according to the existense of model file
if not os.path.exists(prediction_model_file):
    # resample data set
    fdata = []
    colKey = 0
    col = []
    with open(dataset_file, 'rb') as f:
        for dline in f:
            dline = dline.decode("utf-8").rstrip("\r\n").split(",")
            if colKey == 0:
                col = dline
            else:
                fdata.append(dline)
            colKey += 1

    data = pd.DataFrame(fdata, columns = col)

    # data = pd.read_csv(dataset_file, encoding="gb2312")
    data = data.sample(frac=1)
    print(data.head())
    print(data.shape)

    data = data[pd.notnull(data['text'])]
    data = data[pd.notnull(data['label'])]
    data = data.drop(data.columns[0], axis=1)

    # 中文分词
    def jieba_cut(input):
        res = jieba.cut(input, cut_all=False, HMM=True)
        res = " ".join(res)
        # print(res)
        return res
    data['chatmsg_cut'] = data.apply(lambda row: jieba_cut(row["text"]), axis=1)
    data['extra'] = 1

    data.to_csv(dataset_resampled_file)
else:
    # get resampled data set
    data = pd.read_csv(dataset_resampled_file)


train_size = int(len(data) * .8)
print("Train size: %d" % train_size)
print("Test size: %d" % (len(data) - train_size))

# Train features
chatmsg_train = data['chatmsg_cut'][:train_size]
extra_train = data['extra'][:train_size]
# Train labels
labels_train_arr = data['label'][:train_size]

# Test features
chatmsg_test = data['chatmsg_cut'][train_size:]
extra_test = data['extra'][train_size:]
# Test labels
labels_test_arr = data['label'][train_size:]

all_chatmsg = data['chatmsg_cut'][:]

all_labels = data['label'][:]

encoder = LabelEncoder()
encoder.fit(all_labels)
labels_train = encoder.transform(labels_train_arr)
labels_test = encoder.transform(labels_test_arr)
num_classes = np.max(labels_train) + 1
labels_train = keras.utils.to_categorical(labels_train, num_classes)
labels_test = keras.utils.to_categorical(labels_test, num_classes)

num_classes = data['label'].nunique()


# WIDE 1. chat msg matrix
vocab_size = 12000          # 词袋数量
tokenize = keras.preprocessing.text.Tokenizer(num_words=vocab_size, char_level=False)
tokenize.fit_on_texts(all_chatmsg)

chatmsg_bow_train = tokenize.texts_to_matrix(chatmsg_train)
chatmsg_bow_test = tokenize.texts_to_matrix(chatmsg_test)
print("chatmsg_bow_train", chatmsg_bow_train)

# DEEP 1. chat msg sequences
train_embed = tokenize.texts_to_sequences(chatmsg_train)
test_embed = tokenize.texts_to_sequences(chatmsg_test)


max_seq_length = 170
train_embed = keras.preprocessing.sequence.pad_sequences(train_embed, maxlen=max_seq_length, padding="post")
test_embed = keras.preprocessing.sequence.pad_sequences(test_embed, maxlen=max_seq_length, padding="post")
print("train_embed", train_embed)

if not os.path.exists(prediction_model_file):
    ################
    ## Wide Model ##
    ################
    bow_inputs = layers.Input(shape=(vocab_size,), dtype='float32', name='input_bow')
    extra_inputs = layers.Input(shape=(1,), dtype='float32', name='input_extra')
    merged_layer = layers.concatenate([bow_inputs, extra_inputs])
    merged_layer = layers.Dense(256, activation='relu')(merged_layer)
    inter_layer = layers.Dense(num_classes)(merged_layer)
    predictions = layers.Activation('softmax')(inter_layer)
    wide_model = keras.Model(inputs=[bow_inputs, extra_inputs], outputs=predictions)

    print("wide_model.summary", wide_model.summary())

    wide_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    ################
    ## Deep Model ##
    ################
    deep_inputs = layers.Input(shape=(max_seq_length,), dtype='int32', name='input_embed')
    embedding = layers.Embedding(vocab_size, 8, input_length=max_seq_length)(deep_inputs)
    embedding = layers.Flatten()(embedding)
    embedding = layers.Dense(num_classes)(embedding)
    embed_out = layers.Activation('softmax')(embedding)
    deep_model = keras.Model(inputs=deep_inputs, outputs=embed_out)
    print("deep_model.summary", deep_model.summary())

    deep_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    ################
    ##### MERGE ####
    ################
    merged_out = layers.concatenate([wide_model.output, deep_model.output])
    merged_out = layers.Dense(num_classes)(merged_out)
    merged_out = layers.Activation('softmax')(merged_out)
    combined_model = keras.Model(wide_model.input + [deep_model.input], merged_out)
    print("combined_model.summary", combined_model.summary())

    combined_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Training
    combined_model.fit([chatmsg_bow_train, extra_train] + [train_embed], labels_train, epochs=50, batch_size=128)

    # Evaluation
    score = combined_model.evaluate([chatmsg_bow_test, extra_test] + [test_embed], labels_test, batch_size=128)
    print("%s: %.2f%%" % (combined_model.metrics_names[1], score[1] * 100))

    combined_model.save(prediction_model_file)
else:
    combined_model = keras.models.load_model(prediction_model_file)


# predict
# print([description_bow_test, price_test])
predictions = combined_model.predict([chatmsg_bow_test, extra_test] + [test_embed])

num_predictions = len(predictions)

res = []
for i in range(num_predictions):
    prediction = predictions[i]
    index = np.argmax(prediction)
    label_name = encoder.inverse_transform(index)

    print('chatmsg: ', chatmsg_test.iloc[i], '\nindex: ', index, 'Predicted: ', label_name, 'Actual: ', labels_test_arr.iloc[i], '\n')

    res.append([chatmsg_test.iloc[i], index, label_name, labels_test_arr.iloc[i]])

respd = pd.DataFrame(res, columns=["chatmsg_cut", "index", "label_name", "Actual"])
respd.to_csv("dataset/res.csv")