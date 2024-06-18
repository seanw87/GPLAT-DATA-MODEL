import warnings
warnings.filterwarnings("ignore")
import sys, os, tqdm
import pandas as pd
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

sys.path.append('./socialContentParticipial/')
from socialContentParticipial import common

data_file_path = "data/res2.csv"

def docs_accumulatee_by_user(data, corpus_col):
    """
    获得文档数据，根据用户汇总
    :param data:
    :param corpus_col:
    :return:
    """
    user_array = {}
    tmp = 0
    for i, row in data.iterrows():
        tmp+=1
        if tmp>= 100000: break
        if row['uid'] not in user_array:
            user_array[row['uid']] = str(row[corpus_col])
        else:
            user_array[row['uid']] += (' ' + str(row[corpus_col]))

    corpus = []
    for uid in user_array:
        corpus.append(user_array[uid])
    return corpus


if __name__ == "__main__":
    stopwords = common.load_stop_words()

    participialed_res = common.load_data(data_file_path)
    # print(participialed_res)

    # doc = common.get_corpus(participialed_res, 'info_participialed')
    doc = docs_accumulatee_by_user(participialed_res, 'info_participialed')
    print(doc)

    # vectorizer = CountVectorizer()
    # transformer = TfidfTransformer()
    # tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))

    # word = tfidf_model.get_feature_names()  # 所有文本的关键字
    # weight = tfidf.toarray()  # 对应的tfidf矩阵


    tfidf_model = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b",
                                  stop_words=stopwords,
                                  max_features = 20,
                                  # vocabulary={"上课":0, "上学":1,"学校":2,"放假":3,"小学生":4,"写作业":5,"高考":6,"同学":7}
                              ).fit(doc)
    print(tfidf_model.vocabulary_)
    sparse_result = tfidf_model.transform(doc)
    print(sparse_result)



    # data_df = pd.DataFrame(tfidf.toarray())
    # data_df.to_csv("data/tfidf_matrix.csv")



