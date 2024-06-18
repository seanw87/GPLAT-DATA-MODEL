from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow import keras

prediction_simple_model_file = "model/community_auto_tag_simple.h5"
prediction_cnn1d_model_file = "model/community_auto_tag_cnn1d.h5"

dataset_clean_file = "dataset/posts/posts_unlabeled_cleaned.csv"
posts_unlabeled_predicted = "dataset/posts/posts_unlabeled_predicted.csv"
dataset_labels_file = "dataset/posts/labels.csv"

labels = pd.read_csv(dataset_labels_file)
data = pd.read_csv(dataset_clean_file)

simple_model = keras.models.load_model(prediction_simple_model_file)
cnn_model = keras.models.load_model(prediction_cnn1d_model_file)


df_res = pd.DataFrame()
content_cut_arr = []
labels_simple_arr = []
labels_cnn1d_arr = []

for content_cut in data["content_cut"].astype(str):
    content_cut_arr.append(content_cut)

    vocab_size = 12000
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size, char_level=False)
    tokenizer.fit_on_texts(content_cut)

    embed = tokenizer.texts_to_sequences(content_cut)
    max_seq_length = 170
    x = keras.preprocessing.sequence.pad_sequences(embed, maxlen=max_seq_length, padding="post")

    res_simple = simple_model.predict(x)
    # print(res_simple[0])

    # class_res_simple = list(zip(labels['class'], res_simple[0]))
    df_simple = pd.DataFrame()
    df_simple['label'] = labels['class']
    df_simple['score'] = res_simple[0]
    score_sorted_simple = df_simple.sort_values(by='score', ascending=False)[:3]
    labels_simple = score_sorted_simple['label'].values.tolist()
    labels_str_simple = ",".join(str(item) for item in labels_simple)
    labels_simple_arr.append(labels_str_simple)


    res_cnn1d = cnn_model.predict(x)
    # print(res_cnn1d[0])

    # class_res_cnn1d = list(zip(labels['class'], res_cnn1d[0]))
    df_cnn1d = pd.DataFrame()
    df_cnn1d['label'] = labels['class']
    df_cnn1d['score'] = res_cnn1d[0]
    score_sorted_cnn1d = df_cnn1d.sort_values(by='score', ascending=False)[:3]
    labels_cnn1d = score_sorted_cnn1d['label'].values.tolist()
    labels_str_cnn1d = ",".join(str(item) for item in labels_cnn1d)
    labels_cnn1d_arr.append(labels_str_cnn1d)

df_res["content_cut"] = content_cut_arr
df_res["labels_simple"] = labels_simple_arr
df_res["labels_cnn1d"] = labels_cnn1d_arr
df_res.to_csv(posts_unlabeled_predicted)