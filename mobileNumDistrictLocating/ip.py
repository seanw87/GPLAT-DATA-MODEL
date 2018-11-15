# coding: utf-8

from urllib import request
from bs4 import BeautifulSoup
from socket import timeout
import csv, socket

dataset_dir = "dataset/"

def IPspider(numpage, threadId):
    csvfile = open(dataset_dir + 'ips_' + threadId + '.csv', 'wb')
    url = 'http://www.xicidaili.com/nn/'
    user_agent = 'IP'
    headers = {'User-agent': user_agent}
    for num in range(1, numpage + 1):
        ipurl = url + str(num)
        print('Now downloading the ' + str(num * 100) + ' ips')
        req = request.Request(ipurl, headers=headers)
        html = request.urlopen(req).read()
        bs = BeautifulSoup(html, 'html.parser')
        res = bs.find_all('tr')
        for item in res:
            try:
                temp = []
                tds = item.find_all('td')
                temp.append(tds[1].text.encode("utf-8"))
                temp.append(tds[2].text.encode("utf-8"))
                item = (tds[1].text + "," + tds[2].text + "\n").encode("utf-8")
                csvfile.write(item)
            except IndexError:
                pass
    csvfile.close()


def IPpool(threadId, verified = True):
    socket.setdefaulttimeout(2)
    reader = csv.reader(open(dataset_dir + 'ips_' + threadId + '.csv'))
    proxies = []

    for row in reader:
        proxy = row[0] + ':' + row[1]
        proxy_handler = request.ProxyHandler({"http": proxy})
        opener = request.build_opener(proxy_handler)
        request.install_opener(opener)
        try:
            if verified:
                request.urlopen('https://www.baidu.com')
            proxies.append("http://" + row[0] + ":" + row[1])
            print('IPpool append, proxy: ' + proxy)
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

    return proxies