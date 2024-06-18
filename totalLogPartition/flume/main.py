#/bin/python

import pandas as pd

# 批量生成channel和sink配置
csv_data = pd.read_csv("../data/total_log_keywords", delimiter= "|", header = None, names = ['no', 'keyword'],  encoding = "utf-8")
# for number in range(csv_data['no']):
#     print(number)
#     # kw = csv_data['keyword'][number]
#     # print(number, kw)
#     print(csv_data['keyword'])

for i, j in csv_data.iterrows():
    # print(j['no'], j['keyword'])
    template = """
flume4qqdz.channels.channel{no}.type = file
flume4qqdz.channels.channel{no}.checkpointDir = /home/qqdz/flume4qqdz_channel/checkpoint/{keyword}
flume4qqdz.channels.channel{no}.dataDirs = /home/qqdz/flume4qqdz_channel/data/{keyword}
flume4qqdz.channels.channel{no}.transactionCapacity = 10000
flume4qqdz.channels.channel{no}.capacity = 1000000
    """

    print(template.format(no=j['no'], keyword=j['keyword']))

    template2 = """
flume4qqdz.sinks.sink{no}.type = com.ztgame.flume.sink.fileroll.RollingFileSink
flume4qqdz.sinks.sink{no}.channel = channel{no}
flume4qqdz.sinks.sink{no}.sink.directory = /home/qqdz/flume4qqdz_sink/mod={keyword}
flume4qqdz.sinks.sink{no}.sink.minuteRound = 60
flume4qqdz.sinks.sink{no}.sink.rollInterval = 0
    """
    # print(template2.format(no=j['no'], keyword=j['keyword']))
