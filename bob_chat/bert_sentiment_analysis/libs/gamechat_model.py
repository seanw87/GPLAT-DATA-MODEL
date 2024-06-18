#!/usr/bin/env python
# -*- coding:utf-8 -*-

import time, datetime, json, sys, io
import pandas as pd
import numpy as np
# import timeutil

sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8') #

# jieba_dict_file = "/home/qqdz/dmp-crontab/dataset/jieba-dict/dict.txt"
# jieba.load_userdict(jieba_dict_file)

sentiments = {
    0: "正常聊天",
    1: "异常聊天"
}

def getDiffDays():
    today = datetime.date.today()
    oneday = datetime.timedelta(days=1)
    someday = today - oneday
    return someday
def getYesterdayMonth():
    someday = getDiffDays()
    yesterday_month = someday.strftime("%m")
    yesterday_month = str(int(yesterday_month))
    yesterday_month_fmt = someday.strftime("%Y") + "." + yesterday_month
    return yesterday_month_fmt
def getYesterday():
    someday = getDiffDays()
    yesterday = someday.strftime("%Y.%m.%d")
    return yesterday
def getYesterdayTs():
    someday = getDiffDays()
    yesterday_ts = time.mktime(time.strptime(someday.strftime("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S"))
    return yesterday_ts

gamechat_file = "/home/qqdz/dmp-crontab/bob_chat/data/test_raw.csv"
gamechat_res_dir = "/home/qqdz/dmp-crontab/bob_chat/data"


chatresult = None
def get_yesterday_gamechat():
    global chatresult

    col_names = ['id', 'ts', 'uid', 'room_id_uid', 'infos', 'msg_raw', 'type_id', 'init_label']
    dict_dtypes = {x: 'str' for x in col_names}

    gamechat_df = pd.read_csv(gamechat_file, encoding='utf-8',
                              names=col_names,
                              dtype=str,
                              error_bad_lines=False)
    # gamechat_df.columns=['id', 'ts', 'uid', 'room_id_uid', 'infos', 'msg_raw', 'type_id']

    chatresult = [list(a) for a in zip(gamechat_df['id'].tolist(), gamechat_df['ts'].tolist(),
                                       gamechat_df['uid'].tolist(), gamechat_df['room_id_uid'].tolist(),
                                       gamechat_df['infos'].tolist(), gamechat_df['msg_raw'].tolist(),
                                       gamechat_df['type_id'].tolist(), gamechat_df['init_label'].tolist())]
    # print(chatresult)

    predict_literal = [list(a) for a in zip(gamechat_df['init_label'].tolist(), gamechat_df['msg_raw'].tolist())]
    predict_literal = [['init_label', 'msg_raw']] + predict_literal
    # print(predict_literal)

    # yesterday_start_ts = getYesterdayTs()
    # yesterday_end_ts = yesterday_start_ts + 24*3600 - 1

    # predict_literal = [["ID", "desc"]]
    # predict_literal = [['ID', 'desc'], ['0', '你们官方的真的一点良心都没有，如果还有一点的话，就不要再打压我了，求你们给点东西吧'],
    #       ['0', '飞行棋抽奖积分明明没有动却自动清零了，活动说明结束才回清零'],
    #       ['0', '怎么举报好友，有人骗我菜都'],
    #       ['0', '为什么我进不去团战，一直转圈圈'],
    #       ['0', '礼包买不了啊'],
    #       ['0', '真的，我做什么事情都不顺。凭什么别人运气那么好，而我屁也没有。每次都这样，我真的不想玩了，我要退邮永远爪玩了。'
    #             '我会让别人也别玩，如果再这样下去的话，你们球球大作战迟早会没人玩？，希望你能让我好一点，不要总是被官方打压'],
    #       ['0', '建议封一个人号：顾潜  双冠代号'],
    #       ['0', "手机4'g网络为什么打不了这游戏"],
    #       ['0', '你马的垃圾球球，你特么快点倒闭去吧，我特么玩了这个游戏五年，可是我一玩大逃杀就卡，真的是透心凉了，呵呵，'
    #             '退游了，还天天要氪金，快点倒闭吧'],
    #       ['0', '我比第100名的测涨分高，为什么我没有排名？']]
    # esresult = [['player', '', '', '', '你们官方的真的一点良心都没有，如果还有一点的话，就不要再打压我了，求你们给点东西吧', '', 'HUAWEI ASK-AL00x|Android OS 9 / API-28 (HONORASK-AL00x/9.1.1.151C00)', '12.0.5(70011845)', '5e7a37c4ad1968e8b78b50f2', '112.21.245.235', '10223648', '', '', 1585067971, '沈之阳', 1, '24618032', '你们|官方|的|真的|一点|良心|都|没有|，|如果|还有|一点|的话|，|就|不要|再|打压|我|了|，|求|你们|给|点|东西|吧'], ['player', '', '', 'QQ:26081284', '飞行棋抽奖积分明明没有动却自动清零了，活动说明结束才回清零', '', 'HUAWEI LON-AL00|Android OS 9 / API-28 (HUAWEILON-AL00/9.1.0.212C00)', '12.0.5(70011845)', '5e7a3784ac1968d81a8b4eb8', '59.42.185.165', '10223681', '', '', 1585067908, '明寒语', 1, '98733216', '飞行棋|抽奖|积分|明明|没有|动|却|自动|清零|了|，|活动|说明|结束|才|回|清零'], ['player', '', '', '', '怎么举报好友，有人骗我菜都', '', 'vivo V1911A|Android OS 9 / API-28 (PKQ1.181030.001/compiler02261735)', '12.0.5(70011845)', '5e7a376ead1968c7b48b50f2', '117.136.21.25', '10223652', '', '', 1585067886, '触手顶尖迷人花', 1, '58049359', '怎么|举报|好友|，|有人|骗|我|菜|都'], ['player', '', '', 'QQ:3065739935Tel:15328683540', '为什么我进不去团战，一直转圈圈', '', 'iPhone82|iOS 13.1.2', '12.1.1(700121106)', '5e7a3720ac1968af1a8b4eb9', '222.213.250.50', '10223667', '', '', 1585067808, '可以r', 1, '72207130', '为什么|我|进不去|团战|，|一直|转圈圈'], ['player', '', '', '', '礼包买不了啊', '', 'vivo V1829A|Android OS 9 / API-28 (PKQ1.181030.001/compiler01031512)', '12.1.1(70011845)', '5e7a36b2ac1968231d8b4eb8', '117.155.196.109', '10223660', '', '', 1585067698, 'yzhyzh555', 1, '654459811', '礼包|买|不了|啊'], ['player', '', '', '', '真的，我做什么事情都不顺。凭什么别人运气那么好，而我屁也没有。每次都这样，我真的不想玩了，我要退邮永远爪玩了。我会让别人也别玩，如果再这样下去的话，你们球球大作战迟早会没人玩？，希望你能让我好一点，不要总是被官方打压', '', 'HUAWEI ASK-AL00x|Android OS 9 / API-28 (HONORASK-AL00x/9.1.1.151C00)', '12.0.5(70011845)', '5e7a369fac1968a61c8b4eb8', '112.21.245.235', '10223648', '', '', 1585067679, '沈之阳', 1, '24618032', '真的|，|我|做|什么|事情|都|不顺|。|凭|什么|别人|运气|那么|好|，|而|我|屁|也|没有|。|每次|都|这样|，|我|真的|不想|玩|了|，|我要|退邮|永远|爪|玩|了|。|我会|让|别人|也|别|玩|，|如果|再|这样|下去|的话|，|你们|球球|大|作战|迟早会|没人|玩|？|，|希望|你|能|让|我|好|一点|，|不要|总是|被|官方|打压'], ['player', '', '', '', '建议封一个人号：顾潜  双冠代号', '', 'OPPO OPPO A37t|Android OS 6.0 / API-23 (MRA58K/1572355982)', '12.1.1(70011845)', '5e7a3643ac19682f1c8b4eb8', '39.169.181.193', '10223652', '', '', 1585067587, '江西最nb玩家', 1, '880267376', '建议|封|一个|人号|：|顾潜| | |双冠|代号'], ['player', '', '', '', "手机4'g网络为什么打不了这游戏", '', 'iPhone116|iOS 13.3.1', '12.1.1(700121106)', '5e7a35ebad1968f6b38b50f2', '171.210.239.90', '10223667', '', '', 1585067499, 'Boy嗯哼', 1, '44705647', "手机|4|'|g|网络|为什么|打|不了|这|游戏"], ['player', '', '', '', '你马的垃圾球球，你特么快点倒闭去吧，我特么玩了这个游戏五年，可是我一玩大逃杀就卡，真的是透心凉了，呵呵，退游了，还天天要氪金，快点倒闭吧', '', 'Xiaomi Redmi 7|Android OS 9 / API-28 (PKQ1.181021.001/V11.0.3.0.PFLCNXM)', '12.0.0(70011845)', '5e7a35c5ac19685e1c8b4eb8', '223.104.64.199', '10223660', '', '', 1585067461, '琼_无敌1314', 1, '572919903', '你|马|的|垃圾|球球|，|你|特|么|快点|倒闭|去|吧|，|我特|么|玩|了|这个|游戏|五年|，|可是|我|一玩|大逃杀|就|卡|，|真的|是|透心凉|了|，|呵呵|，|退游|了|，|还|天天|要|氪|金|，|快点|倒闭|吧'], ['player', '{"ErrCode":0,"Data":{"Id":"5e7a34d59d785b4ecde1cc57","Uid":733287378,"FileSize":118515,"ShowNum":0,"LaudNum":0,"UploadUrl":"733287378_1585067219_7054982.jpeg","Thumbnail":{"198x198":"733287378_1585067219_7054983.jpeg"},"Time":1585067219,"State":0,"Account":"","CorpId":0,"CorpName":"","Suggest":""}}', '', 'QQ:2628580738Tel:15378919089', '我比第100名的测涨分高，为什么我没有排名？', '', 'OPPO OPPO A73t|Android OS 7.1.1 / API-25 (N6F26Q/1576025554)', '12.0.5(70011845)', '5e7a34d5ac1968551b8b4eb8', '124.224.34.222', '10223680', '', '', 1585067221, '孤殇最菜', 1, '733287378', '我|比|第|100|名|的|测涨|分高|，|为什么|我|没有|排名|？']]
    return predict_literal

def update_yesterday_gamechat(result_list, task_name):
    global chatresult

    for i, item in enumerate(chatresult):
        keyws = chatresult[i].pop()
        chatresult[i].append(result_list[i])
        chatresult[i].append(keyws)

        # index = {
        #     "update": {"_index": es_index, "_type": "report", "_id": 'ObjectIdHex("'+item[8]+'")'}
        # }
        # act = {
        #     "doc": {"zsentiment": sentiments.get(result_list[i])}
        # }
        # upacts += json.dumps(index) + "\n" + json.dumps(act) + "\n"

    #print("upacts: " + upacts)
    # success = es.bulk(upacts, index=es_index, doc_type="report")

    df = pd.DataFrame(chatresult)
    df.to_csv(gamechat_res_dir + "/output_" + task_name + "_" + getYesterday() + ".csv", header=None, sep="#")




if __name__ == '__main__':
    get_yesterday_gamechat()