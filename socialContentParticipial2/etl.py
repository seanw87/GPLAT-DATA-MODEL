import warnings
warnings.filterwarnings("ignore")
import os, sys, re
import jieba
import pandas as pd
sys.path.append('./socialContentParticipial/')
from socialContentParticipial import common

data_file_path = "data/qqdz_test_qjn_teen_chat.csv"

# jieba_dict_file = "data/jieba/dict.txt"
# jieba.load_userdict(jieba_dict_file)

# # 中文分词
# def jieba_cut(sentence):
#     res = jieba.cut(sentence, cut_all=False, HMM=True)
#     res = "|".join(res)
#     return str(res)
#
# def load_data():
#     data_df = pd.read_csv(data_file_path, encoding='utf-8')
#     return data_df

def literal_participial(data_df):
    res = []
    # tmp = 0
    for i, row in data_df.iterrows():
        # print(row['info'])
        if re.search('\.amr', str(row['info'])) is not None:
            continue

        row['info_participialed'] = common.jieba_cut(str(row['info']))
        res.append(row)

        # tmp+= 1
        # if tmp >= 10000:
        #     break
    return res



if __name__ == "__main__":
    data_df = common.load_data(data_file_path)
    res = literal_participial(data_df)
    respd = pd.DataFrame(res, columns=["id", "uid", "info", "dates", "type_id", "age", "info_participialed"])
    respd.to_csv("data/res2.csv")





