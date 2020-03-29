from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import jieba
import numpy as np
import pandas as pd

dataset_file = "dataset/posts/posts_labeled.csv"
dataset_predict_file = "dataset/posts/posts_unlabeled.csv"
dataset_resampled_file = "dataset/posts/posts_labeled_resampled.csv"
dataset_predict_clean_file = "dataset/posts/posts_unlabeled_cleaned.csv"
jieba_dict_file = "dataset/jieba-dict/dict.txt"

jieba.load_userdict(jieba_dict_file)

def etl():
    data = pd.read_csv(dataset_file)

    data = data[pd.notnull(data['labelID'])]
    data = data[pd.notnull(data['content'])]
    data = data.drop(data.columns[0], axis=1)
    print("data.head: ", data.head())
    print("data.shape: ", data.shape)

    def clean_and_cut(content):
        # 1. 去除html tag
        clean_regex = re.compile('<.*?>')
        content = re.sub(clean_regex, "", content)
        # 1. 去除话题标签
        clean_regex = re.compile('#.*?#')
        content = re.sub(clean_regex, "", content)
        # print("raw: ", content)
        # 2. jieba.cut
        res = jieba.cut(content, cut_all=False, HMM=True)
        res = ' '.join(res)
        res = res.strip().strip('﻿ ')
        clean_regex = re.compile(r'\s+')
        res = re.sub(clean_regex, ' ', res)
        # print("cutted: ", res)
        return str(res)
    data["content_cut"] = data.apply(lambda row: clean_and_cut(row["content"]), axis=1)

    data = data.drop(['content'], axis=1)
    data = data[pd.notnull(data['content_cut'])]
    data.sample(frac=1)
    data.to_csv(dataset_resampled_file)


    data_pred = pd.read_csv(dataset_predict_file)

    data_pred = data_pred[pd.notnull(data_pred['content'])]
    data_pred = data_pred.drop(data_pred.columns[0], axis=1)
    print("data.head: ", data_pred.head())
    print("data.shape: ", data_pred.shape)
    data_pred["content_cut"] = data_pred.apply(lambda row: clean_and_cut(row["content"]), axis=1)
    data_pred = data_pred.drop(['content'], axis=1)
    data_pred = data_pred[pd.notnull(data_pred['content_cut'])]
    data_pred.to_csv(dataset_predict_clean_file)

if __name__ == "__main__":
    etl()