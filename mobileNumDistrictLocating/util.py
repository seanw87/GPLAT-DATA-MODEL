import random, requests, re, json, time, os
import ip


dataset_dir = "dataset/"

def get_user_agents():
    user_agents = []
    for client in ["Chrome", "Firefox", "Mozilla", "Safari"]:
        with open(dataset_dir + "user-agent/" + client + ".txt") as f:
            user_agents = user_agents + f.read().splitlines()
    return user_agents

user_agents = get_user_agents()
def get_header():
    if len(user_agents) == 0:
        print("ERROR Len of user_agents is 0")
        exit(0)
    return {'User-Agent': random.choice(user_agents)}

def get_proxy(proxies):
    if len(proxies) == 0:
        print("ERROR Len of proxies is 0")
        exit(0)
    proxy = random.choice(proxies)
    return {'http': proxy}, proxy

def get_cookies():
    cookies = dict(_tb_token_="f3b1e87551767", cookie2="18f0ffef1781c019cea0318cc17ae7ba",
                   t="83f1014f21c7fefb10be1f7fc5eba3f8", v="0")
    return cookies


def jsonobj2csvrow(rjson):
    telString = rjson["telString"] if "telString" in rjson else ""
    catName = rjson["catName"] if "catName" in rjson else ""
    province = rjson["province"] if "province" in rjson else ""
    areaVid = rjson["areaVid"] if "areaVid" in rjson else ""
    carrier = rjson["carrier"] if "carrier" in rjson else ""
    ispVid = rjson["ispVid"] if "ispVid" in rjson else ""
    mts = rjson["mts"] if "mts" in rjson else ""
    data_row = telString + "," + catName + "," + province + "," + areaVid + "," + carrier + "," + \
               ispVid + "," + mts + "\n"
    return data_row

def request_attempt(proxies, mobile_num, res_handler, threadId):
    status = 0
    proxy, proxy_raw = get_proxy(proxies)
    try:
        r = requests.get("https://tcc.taobao.com/cc/json/mobile_tel_segment.htm?tel=" + mobile_num,
                         proxies = proxy,
                         headers = get_header(),
                         cookies = get_cookies())
        rtext = r.text.split("=")
        if len(rtext) > 1:
            json_raw = rtext[1].strip().replace("'", "\"")
            json_raw = re.sub(r"([a-zA-Z]+):", "\"\g<1>\":", json_raw)
            rjson = json.loads(json_raw)
            data_row = jsonobj2csvrow(rjson)
            res_handler.write(data_row.encode("utf-8"))
    except ConnectionRefusedError as e:
        print("ConnectionRefusedError, mobile_phone: " + mobile_num, e)
        status = -1
    except ConnectionAbortedError as e:
        print("ConnectionAbortedError, mobile_phone: " + mobile_num, e)
        status = -2
    except ConnectionResetError as e:
        print("ConnectionResetError, mobile_phone: " + mobile_num, e)
        status = -3
    except requests.exceptions.Timeout as e:
        print("requests.exceptions.Timeout, mobile_phone: " + mobile_num, e)
        status = -4
    except requests.exceptions.TooManyRedirects as e:
        print("requests.exceptions.TooManyRedirects, mobile_phone: " + mobile_num, e)
        status = -5
    except requests.exceptions.RequestException as e:
        print("requests.exceptions.RequestException, mobile_phone: " + mobile_num, e)
        status = -6
    except Exception as e:
        print("Exception, mobile_phone: " + mobile_num, e)
        return False, []

    if status < 0:
        if len(proxies) < 10:
            print("Thread " + threadId + " available num of proxies is below 10, start refetching...")
            os.remove(dataset_dir + 'ips_' + threadId + '.csv')
            ip.IPspider(1, threadId)
            new_proxies = ip.IPpool(threadId)
            print("Thread " + threadId + " num of proxies: " + str(len(new_proxies)) + ". Continue Crawling...")
        else:
            # drop cur proxy
            proxy_index = proxies.index(proxy_raw)
            new_proxies = proxies[:proxy_index-1] + proxies[proxy_index:]

        return False, new_proxies

    time.sleep(random.randint(100, 300)/1000)
    return True, []