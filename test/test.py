#!/usr/bin/env python
# -*- coding:utf-8 -*-

import requests
import pandas as pd
import re

# word1 = u"中国"
# word2 = "地球中国"
# print(re.match(word1, word2))
# exit(0)

with open("./data/qqpd_ip_latest7days.txt", mode="r", encoding="utf-8") as f:
    line = f.readline()

    result = []
    while line is not None and line != "":
        ip = line.strip()
        print(ip)
        addr = ""
        country = ""
        province = ""
        city = ""
        district = ""
        if len(ip) > 0:
            res = requests.get(url="http://127.0.0.1:2060", params={"ip": ip})
            res_data = res.json()
            print(res_data)
            if res_data is not None:
                addr = res_data[ip]['country']

                PATTERN = u'([\u4e00-\u9fa5]{2,5}?(?:省|自治区|市))([\u4e00-\u9fa5]{2,7}?(?:市|区|县|州)){0,1}([\u4e00-\u9fa5]{2,7}?(?:市|区|县)){0,1}'
                pattern = re.compile(PATTERN)
                m = pattern.search(addr)
                if not m:
                    print(country + '|||')
                else:
                    country = '中国'
                    if m.lastindex >= 1:
                        province = m.group(1)
                    if m.lastindex >= 2:
                        city = m.group(2)
                    if m.lastindex >= 3:
                        district = m.group(3)

        result.append([ip, addr, country, province, city, district])
        # print(result)

        line = f.readline()


df = pd.DataFrame(result)
df.to_csv("./data/res_qqpd_ip_latest7days.txt", header=None, sep=",")



