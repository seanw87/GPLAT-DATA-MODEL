# coding: utf-8

from urllib import request
from bs4 import BeautifulSoup
from socket import timeout
import csv, socket
import util

dataset_dir = "dataset/"

def IPspider(numpage, threadId):
    csvfile = open(dataset_dir + 'ips_' + threadId + '.csv', 'wb')
    url = 'http://www.xicidaili.com/nn/'
    headers = util.get_header()
    for num in range(1, numpage + 1):
        ipurl = url + str(num)
        print('Now downloading the ' + str(num * 100) + ' ips for threadId: ' + threadId)
        # 清空代理设置
        proxy_handler = request.ProxyHandler({})
        opener = request.build_opener(proxy_handler)
        request.install_opener(opener)
        req = request.Request(ipurl, headers=headers)
        with request.urlopen(req) as reqobj:
            html = reqobj.read()
            bs = BeautifulSoup(html, 'html.parser')
            res = bs.find_all('tr')
            for item in res:
                try:
                    temp = []
                    tds = item.find_all('td')
                    temp.append(tds[1].text.encode("utf-8"))
                    temp.append(tds[2].text.encode("utf-8"))
                    temp.append(tds[5].text.lower().encode("utf-8"))
                    item = (tds[1].text + "," + tds[2].text + "," + tds[5].text.lower() + "\n").encode("utf-8")
                    csvfile.write(item)
                except IndexError:
                    pass
            reqobj.close()
    csvfile.close()


def IPpool(threadId, verified = True):
    socket.setdefaulttimeout(2)
    csvfile = open(dataset_dir + 'ips_' + threadId + '.csv')
    reader = csv.reader(csvfile)
    proxies = []

    try:
        for row in reader:
            proxy = row[0] + ':' + row[1]
            if "https" == row[2]:
                continue
                # proxy_handler = {"http": proxy, "https": proxy}
            else:
                proxy_handler = {"http": proxy}
            proxy_handler = request.ProxyHandler(proxy_handler)
            opener = request.build_opener(proxy_handler)
            request.install_opener(opener)
            try:
                if verified:
                    request.urlopen('https://www.baidu.com')
                proxies.append("http://" + row[0] + ":" + row[1])
                # print('IPpool append, proxy: ' + proxy)
            except request.HTTPError as e:
                print('HTTPError = ' + str(e.code) + " reason: " + str(e.reason) + ', proxy: ' + proxy)
                continue
            except request.URLError as e:
                print('URLError ' + " reason: " + str(e.reason) + ', proxy: ' + proxy)
                continue
            except ConnectionResetError as e:
                print('ConnectionResetError, proxy: ' + proxy)
            except timeout:
                print('socket timeout, proxy: ' + proxy + ', proxy: ' + proxy)
                continue
            else:
                continue
    finally:
        csvfile.close()

    return proxies