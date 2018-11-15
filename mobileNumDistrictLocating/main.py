import os, csv, requests, random, time, json, re, threading, time, math
import ip


class Crawler_Thread (threading.Thread):
    def __init__(self, threadID, name, counter=0):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
    def run(self):
        print("Starting " + self.name + " " + time.ctime(time.time()))
        mobile_location_crawler(self.threadID)
        print("Exiting " + self.name + " " + time.ctime(time.time()))


dataset_dir = "dataset/"
def mobile_location_crawler(threadId):
    threadId = str(threadId)
    if not os.path.exists(dataset_dir + "ips_" + threadId + ".csv"):
        ip.IPspider(1, threadId)

    user_agents = []
    for client in ["Chrome", "Firefox", "Mozilla", "Safari"]:
        with open(dataset_dir + "user-agent/" + client + ".txt") as f:
            user_agents = user_agents + f.read().splitlines()
    def get_header():
        return {'User-Agent':random.choice(user_agents)}

    proxies = ip.IPpool(threadId)
    print("num of proxies: " + str(len(proxies)))
    def get_proxy():
        return {'http': random.choice(proxies)}

    def get_cookies():
        cookies = dict(_tb_token_="f3b1e87551767", cookie2="18f0ffef1781c019cea0318cc17ae7ba",
                       t="83f1014f21c7fefb10be1f7fc5eba3f8", v="0")
        return cookies




    mobile_reader = csv.reader(open(dataset_dir + 'mobile_split_' + threadId + '.csv'))
    res_handler = open(dataset_dir + 'mobile_res_' + threadId + '.csv', 'wb')
    res_handler.write("telString,catName,province,areaVid,carrier,ispVid,mts\n".encode("utf-8"))
    index = 0
    for row in mobile_reader:
        index = index + 1
        if index % 50000 == 0:
            os.remove(dataset_dir + 'ips_' + threadId + '.csv')
        if not os.path.exists(dataset_dir + 'ips_' + threadId + '.csv'):
            ip.IPspider(1, threadId)
            proxies = ip.IPpool(threadId)

        mobile_num = row[1]
        print(get_proxy())
        try:
            r = requests.get("https://tcc.taobao.com/cc/json/mobile_tel_segment.htm?tel=" + mobile_num,
                             proxies = get_proxy(),
                             headers = get_header(),
                             cookies = get_cookies())
            rtext = r.text.split("=")
            if len(rtext) > 1:
                json_raw = rtext[1].strip().replace("'", "\"")
                json_raw = re.sub(r"([a-zA-Z]+):", "\"\g<1>\":", json_raw)
                print(json_raw)
                rjson = json.loads(json_raw)

                telString = rjson["telString"] if "telString" in rjson else  ""
                catName = rjson["catName"] if "catName" in rjson else ""
                province = rjson["province"] if "province" in rjson else ""
                areaVid = rjson["areaVid"] if "areaVid" in rjson else ""
                carrier = rjson["carrier"] if "carrier" in rjson else ""
                ispVid = rjson["ispVid"] if "ispVid" in rjson else ""
                mts = rjson["mts"] if "mts" in rjson else ""
                print(telString + "," + catName + "," + province + "," + areaVid + "," + carrier + "," +
                                   ispVid + "," + mts + "\n")
                res_handler.write((telString + "," + catName + "," + province + "," + areaVid + "," + carrier + "," +
                                   ispVid + "," + mts + "\n").encode("utf-8"))
        except ConnectionRefusedError as e:
            continue
        time.sleep(random.randint(100, 300)/1000)


thread_num = 10
total_fh = open(dataset_dir + 'mobile.csv')
try:
    i = 0
    lines = total_fh.readlines()
    segnum = math.ceil(len(lines) / thread_num)
    while True:
        i += 1
        split_fh = open(dataset_dir + 'mobile_split_' + str(i) + '.csv', 'a')

        start = (i-1) * segnum
        end = i * segnum
        try:
            for line in lines[start:end]:
                split_fh.write(line)
        finally:
            split_fh.close()

        if len(lines) < i * segnum:
            break

    for i in range(thread_num):
        thread = Crawler_Thread(i+1, "Crawler-Thread-" + str(i+1))
        thread.start()

finally:
    total_fh.close()
    print("Exiting Main Thread")