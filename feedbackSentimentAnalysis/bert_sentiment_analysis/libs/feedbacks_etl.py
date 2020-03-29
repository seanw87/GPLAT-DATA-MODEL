#!/usr/bin/env python
import os
import pandas as pd

if __name__ == '__main__':
    path = "../data/bob/"
    pd_all = pd.read_csv(os.path.join(path, "qiuqiu-report-2020.3"), sep='#')
    # pd_all = shuffle(pd_all)
    # print(pd_all.head())
    # print(pd_all.shape)
    # print(pd_all.ix[:, 5])
    pd_all = pd_all[pd.notnull(pd_all.ix[:, 5])]
    feedbacks = pd_all.ix[:, 5].to_frame(name="feedback")
    print(type(feedbacks))
    feedbacks.insert(loc=0, column="ID", value=0)
    print(feedbacks)
    feedbacks.to_csv(os.path.join(path, "test.csv"), index=False, sep='\t')
