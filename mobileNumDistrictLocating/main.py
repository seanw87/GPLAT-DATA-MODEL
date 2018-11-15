import os, csv, requests, random, time, json, re
import ip


dataset_dir = "dataset/"

if not os.path.exists(dataset_dir + "ips.csv"):
    ip.IPspider(1)

user_agents = []
for client in ["Chrome", "Firefox", "Mozilla", "Safari"]:
    with open(dataset_dir + "user-agent/" + client + ".txt") as f:
        user_agents = user_agents + f.read().splitlines()
def get_header():
    return {'User-Agent':random.choice(user_agents)}

proxies = ip.IPpool()
print("num of proxies: " + str(len(proxies)))
def get_proxy():
    return {'http': random.choice(proxies)}

def get_cookies():
    cookies = dict(_tb_token_="f3b1e87551767", cookie2="18f0ffef1781c019cea0318cc17ae7ba",
                   t="83f1014f21c7fefb10be1f7fc5eba3f8", v="0")
    return cookies




mobile_reader = csv.reader(open(dataset_dir + 'mobile.csv'))
res_handler = open(dataset_dir + 'mobile_res_tmp.csv', 'wb')
res_handler.write("telString,catName,province,areaVid,carrier,ispVid,mts\n".encode("utf-8"))
index = 0
for row in mobile_reader:
    index = index + 1
    if index % 50000 == 0:
        os.remove(dataset_dir + "ips.csv")
    if not os.path.exists(dataset_dir + "ips.csv"):
        ip.IPspider(1)
        proxies = ip.IPpool()

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