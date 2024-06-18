import warnings
warnings.filterwarnings("ignore")
import sys, os, re
import pandas as pd
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt

sys.path.append('./socialContentParticipial/')
from socialContentParticipial import common

data_file_path_adult = "data/labeled_raw/adult/qjn_adult_chat.csv"
data_file_path_juvenile = "data/labeled_raw/juvenile/qjn_teen_chat.csv"

train_path = "data/labeled_participialed"
data_sink_path_adult = os.path.join(train_path, "adult/adult.csv")
data_sink_path_juvenile = os.path.join(train_path, "juvenile/juvenile.csv")

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

def content_proc_adult():
    data_df = common.load_data(data_file_path_juvenile)

    res = literal_participial(data_df)
    respd = pd.DataFrame(res, columns=["id", "uid", "info", "info_participialed"])
    respd.to_csv(data_sink_path_adult)

def content_proc_juvenile():
    data_df = common.load_data(data_file_path_adult)

    res = literal_participial(data_df)
    respd = pd.DataFrame(res, columns=["id", "uid", "info", "info_participialed"])
    respd.to_csv(data_sink_path_juvenile)

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
    """
    # ouput:
    # 114872 114872
    # 80410 80410 34462 34462
    # MNB:0.681768(0.003362)
    # CART:0.646437(0.003759)
    # KNN:0.549198(0.029333)
    # LR:0.685773(0.004761)
    # SVM:0.529151(0.004401)
    # 最优 : 0.6847282676284044 使用 {'C': 5}
    # 最优 : 0.682029598308668 使用 {'alpha': 1.5}
    # RF : 0.660788 (0.005860)
    # AB : 0.631128 (0.006965)
    # 最优 : 0.6759731376694441 使用 {'n_estimators': 150}
    # 0.678486448842203
    #               precision    recall  f1-score   support
    #
    #            0       0.70      0.53      0.61     16058
    #            1       0.66      0.80      0.73     18404
    #
    #    micro avg       0.68      0.68      0.68     34462
    #    macro avg       0.68      0.67      0.67     34462
    # weighted avg       0.68      0.68      0.67     34462
    """

    # 停用词
    stopwords = common.load_stop_words()

    # 训练数据分词
    if not os.path.exists(data_sink_path_adult):
        print(data_sink_path_adult + " not exists, generating...")
        content_proc_adult()
    if not os.path.exists(data_sink_path_juvenile):
        print(data_sink_path_juvenile + " not exists, generating...")
        content_proc_juvenile()

    categories = ['adult', 'juvenile']

    # 导入训练数据
    # dataset_train = load_files(container_path=train_path, load_content=True, encoding='UTF-8')  #categories=categories,
    # # print(dataset_train)
    # print(dataset_train.target, dataset_train.target_names)
    # print(dataset_train.data)
    # dataset_train_data_arr = dataset_train.data.split('\r\n')

    x = []
    y = []

    # adult label: 1
    participialed_adult = common.load_data(data_sink_path_adult)
    for i, row in participialed_adult.iterrows():
        x.append(str(row['info_participialed']))
        y.append(adult_label)

    # juvenile label: 0
    participialed_juvenile = common.load_data(data_sink_path_juvenile)
    for i, row in participialed_juvenile.iterrows():
        x.append(str(row['info_participialed']))
        y.append(juvenile_label)

    print('原始样本和标签：', len(x), len(y))

    # 7:3生成训练数据和测试数据
    dataset_train_data, dataset_test_data, dataset_train_target, dataset_test_target = \
        train_test_split(x, y, test_size = 0.3, random_state = 42)

    dataset_train = DataObj(data=dataset_train_data, target=dataset_train_target)
    dataset_test = DataObj(data=dataset_test_data, target=dataset_test_target)
    print('7:3生成训练数据和测试数据：', len(dataset_train.data), len(dataset_train.target),
          len(dataset_test.data), len(dataset_test.target))

    # # 导入评估数据
    # dataset_test = dataset_train

    # 计算词频
    count_vect = CountVectorizer(stop_words=stopwords, decode_error='ignore')
    X_train_counts = count_vect.fit_transform(dataset_train.data)

    # 计算TF-IDF
    tf_transformer = TfidfVectorizer(stop_words=stopwords, decode_error='ignore')
    X_train_counts_tf = tf_transformer.fit_transform(dataset_train.data)
    # print(X_train_counts_tf)

    # 算法评估基准
    '''采用10折交叉验证的方式来比较算法的准确度'''
    num_folds = 10
    seed = 7
    scoring = 'accuracy'
    # 评估算法
    models = {
        'LR': LogisticRegression(),         # 逻辑回归
        'SVM': SVC(),                       # 支持向量机
        'CART': DecisionTreeClassifier(),   # 分类与回归树
        'MNB': MultinomialNB(),             # 朴素贝叶斯分类器
        'KNN': KNeighborsClassifier()       # K近邻算法
    }

    results = []
    for key in models:
        kfold = KFold(n_splits=num_folds, random_state=seed)
        cv_results = cross_val_score(models[key], X_train_counts_tf, dataset_train.target, cv=kfold, scoring=scoring,
                                     n_jobs=-1)
        results.append(cv_results)
        print('%s:%f(%f)' % (key, cv_results.mean(), cv_results.std()))

    # 逻辑回归调参
    '''逻辑回归中的超参数是C，C值越小正则化强度越大'''
    param_grid = {
        'C': [0.1, 5, 13, 15]
    }
    model = LogisticRegression()
    kfold = KFold(n_splits=num_folds, random_state=seed)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid_result = grid.fit(X=X_train_counts_tf, y=dataset_train.target)
    print('最优 : %s 使用 %s' % (grid_result.best_score_, grid_result.best_params_))
    # 朴素贝叶斯分类器调参
    param_grid = {
        'alpha': [0.001, 0.01, 0.1, 1.5]
    }
    model = MultinomialNB()
    kfold = KFold(n_splits=num_folds, random_state=seed)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid_result = grid.fit(X=X_train_counts_tf, y=dataset_train.target)
    print('最优 : %s 使用 %s' % (grid_result.best_score_, grid_result.best_params_))

    # 集成算法
    ensembles = {
        'RF': RandomForestClassifier(),     # 随机森林
        'AB': AdaBoostClassifier()          # Adaboost
    }
    results = []
    for key in ensembles:
        kfold = KFold(n_splits=num_folds, random_state=seed)
        cv_results = cross_val_score(ensembles[key], X_train_counts_tf, dataset_train.target, cv=kfold, scoring=scoring)
        results.append(cv_results)
        print('%s : %f (%f)' % (key, cv_results.mean(), cv_results.std()))
    # 集成算法调参
    param_grid = {
        'n_estimators': [10, 100, 150, 200]
    }
    model = RandomForestClassifier()
    kfold = KFold(n_splits=num_folds, random_state=seed)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid_result = grid.fit(X=X_train_counts_tf, y=dataset_train.target)
    print('最优 : %s 使用 %s' % (grid_result.best_score_, grid_result.best_params_))

    # 生成模型
    model = LogisticRegression(C=5)
    model.fit(X_train_counts_tf, dataset_train.target)
    X_test_counts = tf_transformer.transform(dataset_test.data)
    predictions = model.predict(X_test_counts)
    print(accuracy_score(dataset_test.target, predictions))
    print(classification_report(dataset_test.target, predictions))


