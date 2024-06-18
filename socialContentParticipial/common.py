import warnings
warnings.filterwarnings("ignore")
import sys, os
import jieba
import pandas as pd

stopwords_file = "../feedbackAssociation/data/stopwords_overall.txt"

jieba_dict_file = "data/jieba/dict.txt"
jieba.load_userdict(jieba_dict_file)

# 中文分词
def jieba_cut(sentence):
    res = jieba.cut(sentence, cut_all=False, HMM=True)
    res = " ".join(res)
    return str(res)

def load_data(data_file_path, header=None, names=[], seperator=','):
    if header is None:
        data_df = pd.read_csv(data_file_path, encoding='utf-8', sep=seperator)
    else:
        data_df = pd.read_csv(data_file_path, encoding='utf-8', names=names, header=0, sep=seperator)
    return data_df

def get_corpus(data, corpus_col):
    """
    获得文档数据
    :param data:
    :param corpus_col:
    :return:
    """
    corpus = []
    for i, row in data.iterrows():
        # print(row['info'])
        corpus.append(str(row[corpus_col]))
    print('length of corpus: ', len(corpus))
    return corpus

def load_stop_words():
    """
    加载停用词
    :return:list
    """
    with open(stopwords_file, encoding="utf-8") as f:
        stopwords = f.read().split("\n")
        return stopwords