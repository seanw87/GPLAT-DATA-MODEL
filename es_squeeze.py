#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import time
import pandas as pd

from elasticsearch import Elasticsearch
sys.path.append(os.path.join(os.path.dirname(__file__), "libs"))
import timeutil

#10.20.32.134:9300,10.20.32.135:9300,10.20.32.136:9300,10.20.32.137:9300,10.20.32.138:9300
es = Elasticsearch(hosts=["10.20.32.138:9200"], timeout=5000) 


def get_esindex(start_timestamp):
    """
    read config info
    """
    date_info = time.localtime(float(start_timestamp))
    es_index = "qiuqiu-game-message-" + str(date_info.tm_year) + '.' + str( date_info.tm_mon)
    print(es_index)
    return es_index


def get_diff_message(start_date):
    """
    """
    # start_date = '2017-06-01 00:00:00'
    # end_date = '2017-06-02 00:00:00'

    start_timestamp = timeutil.date2timestamp(start_date)
    end_time = start_timestamp + 86400

    end_timestamp = start_timestamp + 299
    es_index = get_esindex(start_timestamp)

    result = []
    while start_timestamp < end_time:
        q_body = {
#            "query":{
#                "query_string": {
#                    "query": ""
#                }
#            },
            "filter":{
                "and":[
                    {
                        "range":{
                            "time": {
                                "gte": start_timestamp,
                                "lte": end_timestamp,
                                "format": "epoch_second"
                            }
                        }

                    },
                    {   
                        "query": {
                            "bool": {
                                "must_not": {
                                    "term": {"op_user": "robot"}
                                }
                            }
                        }
                    }      
               ]

            }
        }
        # print start_timestamp, end_time
        res = es.search(index=es_index, doc_type="message", body=q_body, size=10000)
        for item in res['hits']['hits']:
            #print item['_source']['text']
            result.append(item['_source'])

        start_timestamp = start_timestamp + 300
        end_timestamp = start_timestamp + 300
    
    df = pd.DataFrame(result)
    print df.head()
    df.to_csv("/tmp/qiuqiu-message-" + start_date.split(" ")[0])



if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "need date info, date：'2017-06-01 00:00:00'"
    else:
        start_date = sys.argv[1]
        get_diff_message(start_date)

