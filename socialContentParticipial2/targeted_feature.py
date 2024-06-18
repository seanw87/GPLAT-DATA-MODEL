import warnings
warnings.filterwarnings("ignore")
import sys, os, re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.append('./socialContentParticipial/')
from socialContentParticipial import common

refered_path_juvenile = "data/merged_juvenile.csv"
refered_path_adult = "data/merged_adult.csv"

if __name__ == "__main__":
    # 停用词
    stopwords = common.load_stop_words()

    refered_juvenile = common.load_data(refered_path_juvenile, header=None, names=[], seperator='|')
    refered_juvenile_filterd = refered_juvenile[(refered_juvenile['refer']==0)]

    docs = []
    for i, row in refered_juvenile_filterd.iterrows():
        docs.append(str(row['content_literal']))

    tfidf_model = TfidfVectorizer(
                                  stop_words=stopwords,
                                  max_features=2000,
                                  ).fit(docs)
    featurenames = tfidf_model.get_feature_names()  # 所有文本的关键字
    respd = pd.DataFrame(featurenames, columns=["features"])
    respd.to_csv("data/features_juvenile.csv")









    refered_adult = common.load_data(refered_path_adult, header=None, names=[], seperator='|')
    refered_adult_filterd = refered_adult[(refered_adult['refer']==1)]

    docs = []
    for i, row in refered_adult_filterd.iterrows():
        docs.append(str(row['content_literal']))

    tfidf_model = TfidfVectorizer(
                                  stop_words=stopwords,
                                  max_features=2000,
                                  ).fit(docs)
    featurenames = tfidf_model.get_feature_names()  # 所有文本的关键字
    respd = pd.DataFrame(featurenames, columns=["features"])
    respd.to_csv("data/features_adult.csv")
