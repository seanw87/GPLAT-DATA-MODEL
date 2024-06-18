import warnings
warnings.filterwarnings("ignore")
import sys, os, re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

sys.path.append('./socialContentParticipial/')
from socialContentParticipial import common

raw_path = "data/labeled_raw"
data_file_train_path_adult = os.path.join(raw_path, "train/adult/adult.csv")
data_file_train_path_juvenile = os.path.join(raw_path, "train/juvenile/juvenile.csv")
data_file_test_path_adult = os.path.join(raw_path, "test/adult/adult.csv")
data_file_test_path_juvenile = os.path.join(raw_path, "test/juvenile/qjn_find_teen_50_20211129.csv")

participialed_path = "data/labeled_participialed"
data_sink_train_path_adult = os.path.join(participialed_path, "train/adult/adult.csv")
data_sink_train_path_juvenile = os.path.join(participialed_path, "train/juvenile/juvenile.csv")
data_sink_test_path_adult = os.path.join(participialed_path, "test/adult/adult.csv")
data_sink_test_path_juvenile = os.path.join(participialed_path, "test/juvenile/qjn_find_teen_50_20211129.csv")

adult_label = 1
juvenile_label = 0


def literal_participial(data_df):
    res = []
    # tmp = 0
    for i, row in data_df.iterrows():
        # print(row['_c1'])
        content_participialed = ''
        contents = str(row['_c1'])
        contents_arr = contents.split('|')
        for item in contents_arr:
            # 排除语音文件
            if re.search('\.amr', item) is not None:
                continue
            item_participialed = common.jieba_cut(item)
            content_participialed += (' ' + item_participialed)

        row['info_participialed'] = content_participialed
        res.append(row)

        # tmp += 1
        # if tmp >= 10000:
        #     break
    return res

def doc_proc(doc_path, columns, sink_path):
    data_df = common.load_data(doc_path)

    res = literal_participial(data_df)
    respd = pd.DataFrame(res, columns=columns)
    respd.to_csv(sink_path, index=True, index_label='id')

def refer(test_set, flag, test_raw):
    X_test_counts = tf_transformer.transform(test_set.data)
    predictions = model.predict(X_test_counts)
    print('accuracy_score: '+flag+': ', accuracy_score(test_set.target, predictions))
    print('classification_report: '+flag+': ', classification_report(test_set.target, predictions))

    predictions_pd = pd.DataFrame({'key': range(len(predictions)), 'refer': predictions})
    dataset_test_data_pd = pd.DataFrame(
        {'key': range(len(test_set.data)), 'content_literal': test_set.data}
    )

    refer_merge_pd = pd.merge(predictions_pd, dataset_test_data_pd, on=['key'])
    refer_merge_pd.to_csv('data/refer_'+flag+'.csv', sep='|', index=False, encoding='utf-8')

    test_raw.to_csv('data/test_participialed_'+flag+'.csv', sep='|', index=False, encoding='utf-8')
    test_raw['key'] = range(len(test_raw))

    merge_pd = pd.merge(left=refer_merge_pd, right=test_raw, how='inner',
                        on=['key'],
                        left_index=False, right_index=False, indicator=False
                        )
    merge_pd.to_csv('data/merged_'+flag+'.csv', sep='|', index=False, encoding='utf-8')


class DataObj(dict):
    """
    对象类用于保存训练/测试样本和label
    """
    def __init__(self, **kwargs):
        super(DataObj, self).__init__(kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)


if __name__ == "__main__":
    # 停用词
    stopwords = common.load_stop_words()

    # 训练数据分词
    if not os.path.exists(data_sink_train_path_adult):
        print(data_sink_train_path_adult + " not exists, generating...")
        # content_proc_adult()
        doc_proc(data_file_train_path_adult, ["uid", "info", "info_participialed"], data_sink_train_path_adult)
    if not os.path.exists(data_sink_train_path_juvenile):
        print(data_sink_train_path_juvenile + " not exists, generating...")
        # content_proc_juvenile()
        doc_proc(data_file_train_path_juvenile, ["uid", "info", "info_participialed"], data_sink_train_path_juvenile)

    # 测试数据分词
    if not os.path.exists(data_sink_test_path_adult):
        print(data_sink_test_path_adult + " not exists, generating...")
        doc_proc(data_file_test_path_adult, ["uid", "info", "info_participialed"], data_sink_test_path_adult)
    if not os.path.exists(data_sink_test_path_juvenile):
        print(data_sink_test_path_juvenile + " not exists, generating...")
        doc_proc(data_file_test_path_juvenile, ["uid", "info", "info_participialed"], data_sink_test_path_juvenile)

    categories = ['adult', 'juvenile']

    # 导入训练数据 deprecated... load_files只适用于读入整个文件原始文本作为一整个文档的情况，没法分行存入数组
    # dataset_train = load_files(container_path=train_path, load_content=True, encoding='UTF-8')  #categories=categories,
    # # print(dataset_train)
    # print(dataset_train.target, dataset_train.target_names)
    # print(dataset_train.data)
    # dataset_train_data_arr = dataset_train.data.split('\r\n')

    # 训练数据
    x = []; y = []

    # adult label: 1
    train_participialed_adult = common.load_data(data_sink_train_path_adult, header=0, names=['id', 'uid', 'info', 'info_participialed'])
    for i, row in train_participialed_adult.iterrows():
        x.append(str(row['info_participialed']))
        y.append(adult_label)

    # juvenile label: 0
    train_participialed_juvenile = common.load_data(data_sink_train_path_juvenile, header=0, names=['id', 'uid', 'info', 'info_participialed'])
    for i, row in train_participialed_juvenile.iterrows():
        x.append(str(row['info_participialed']))
        y.append(juvenile_label)

    print('训练样本和标签：', len(x), len(y))
    dataset_train_data = x
    dataset_train_target = y

    # 测试数据
    x = []; y = []
    # adult label: 1
    participialed_adult = common.load_data(data_sink_test_path_adult, header=0,
                                           names=['id', 'uid', 'info', 'info_participialed'])
    for i, row in participialed_adult.iterrows():
        x.append(str(row['info_participialed']))
        y.append(adult_label)

    print('测试样本和标签（成年）：', len(x), len(y))
    dataset_test_data_adult = x
    dataset_test_target_adult = y

    x = []; y = []
    # juvenile label: 0
    participialed_juvenile = common.load_data(data_sink_test_path_juvenile, header=0,
                                              names=['id', 'uid', 'info', 'info_participialed'])
    for i, row in participialed_juvenile.iterrows():
        x.append(str(row['info_participialed']))
        y.append(juvenile_label)

    print('测试样本和标签（未成年）：', len(x), len(y))
    dataset_test_data_juvenile = x
    dataset_test_target_juvenile = y


    # 生成训练数据和测试数据
    # dataset_train_data, dataset_test_data, dataset_train_target, dataset_test_target = \
    #     train_test_split(x, y, test_size = 0.3, random_state = 42)
    dataset_train = DataObj(data=dataset_train_data, target=dataset_train_target)
    dataset_test_adult = DataObj(data=dataset_test_data_adult, target=dataset_test_target_adult)
    dataset_test_juvenile = DataObj(data=dataset_test_data_juvenile, target=dataset_test_target_juvenile)
    print('训练数据和测试数据大小。',
          '训练样本：', len(dataset_train.data), '训练标签：', len(dataset_train.target),
          '测试样本（成年）：', len(dataset_test_adult.data), '测试标签（成年）：', len(dataset_test_adult.target),
          '测试样本（未成年）：', len(dataset_test_juvenile.data), '测试标签（未成年）：', len(dataset_test_juvenile.target)
          )

    # 计算词频
    count_vect = CountVectorizer(stop_words=stopwords, decode_error='ignore')
    X_train_counts = count_vect.fit_transform(dataset_train.data)

    # 计算TF-IDF
    tf_transformer = TfidfVectorizer(stop_words=stopwords, decode_error='ignore')
    X_train_counts_tf = tf_transformer.fit_transform(dataset_train.data)
    # print(X_train_counts_tf)

    # 生成模型
    model = LogisticRegression(C=5)
    model.fit(X_train_counts_tf, dataset_train.target)




    # 推理（成年）
    # refer(dataset_test_adult, 'adult', participialed_adult)

    # X_test_counts_adult = tf_transformer.transform(dataset_test_adult.data)
    # predictions = model.predict(X_test_counts_adult)
    # print('accuracy_score: adult: ', accuracy_score(dataset_test_adult.target, predictions))
    # print('classification_report: adult: ', classification_report(dataset_test_adult.target, predictions))
    #
    # predictions_pd = pd.DataFrame({'key':range(len(predictions)), 'refer':predictions})
    # dataset_test_data_pd = pd.DataFrame({'key':range(len(dataset_test_adult.data)), 'content_literal': dataset_test_adult.data})
    #
    # refer_merge_pd = pd.merge(predictions_pd, dataset_test_data_pd, on=['key'])
    # refer_merge_pd.to_csv('data/refer_adult.csv', sep='|', index=False, encoding='utf-8')
    #
    # participialed_adult.to_csv('data/test_participialed_adult.csv', sep='|', index=False, encoding='utf-8')
    # participialed_adult['key'] = range(len(participialed_adult))
    #
    # merge_pd = pd.merge(left=refer_merge_pd, right=participialed_adult, how='inner',
    #                     on=['key'],
    #                     left_index=False, right_index=False, indicator=False
    #                     )
    # # merge_pd = merge_pd.drop(['key', 'id'], axis=1).drop_duplicates()
    # merge_pd.to_csv('data/merged_adult.csv', sep='|', index=False, encoding='utf-8')





    # 推理（未成年）
    refer(dataset_test_juvenile, 'juvenile', participialed_juvenile)

    # X_test_counts_juvenile = tf_transformer.transform(dataset_test_juvenile.data)
    # predictions = model.predict(X_test_counts_juvenile)
    # print('accuracy_score: juvenile: ', accuracy_score(dataset_test_juvenile.target, predictions))
    # print('classification_report: juvenile: ', classification_report(dataset_test_juvenile.target, predictions))
    #
    # predictions_pd = pd.DataFrame({'key':range(len(predictions)), 'refer':predictions})
    # dataset_test_data_pd = pd.DataFrame({'key':range(len(dataset_test_juvenile.data)), 'content_literal': dataset_test_juvenile.data})
    #
    # refer_merge_pd = pd.merge(predictions_pd, dataset_test_data_pd, on=['key'])
    # refer_merge_pd.to_csv('data/refer_juvenile.csv', sep='|', index=False, encoding='utf-8')
    #
    # participialed_juvenile.to_csv('data/test_participialed_juvenile.csv', sep='|', index=False, encoding='utf-8')
    # participialed_juvenile['key'] = range(len(participialed_juvenile))
    #
    # merge_pd = pd.merge(left=refer_merge_pd, right=participialed_juvenile, how='inner',
    #                     on=['key'],
    #                     left_index=False, right_index=False, indicator=False
    #                     )
    # # merge_pd = merge_pd.drop(['key', 'id'], axis=1).drop_duplicates()
    # merge_pd.to_csv('data/merged_juvenile.csv', sep='|', index=False, encoding='utf-8')