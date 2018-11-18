import os, csv, threading, time, math
import ip, util


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

    proxies = ip.IPpool(threadId)
    print("Thread " + threadId + " num of proxies: " + str(len(proxies)) + ". Start Crawling...")


    mobile_reader = csv.reader(open(dataset_dir + 'mobile_split_' + threadId + '.csv'))
    res_handler = open(dataset_dir + 'mobile_res_' + threadId + '.csv', 'wb')
    res_handler.write("telString,catName,province,areaVid,carrier,ispVid,mts\n".encode("utf-8"))
    index = 0
    for row in mobile_reader:
        index = index + 1
        if index % 50000 == 0:
            print("Thread " + threadId + " num of request equal 50000, start refetching proxies...")
            os.remove(dataset_dir + 'ips_' + threadId + '.csv')
            ip.IPspider(1, threadId)
            proxies = ip.IPpool(threadId)
            print("Thread " + threadId + " num of proxies: " + str(len(proxies)) + ". Continue Crawling...")

        mobile_num = row[1]
        while True:
            ret, updated_proxies = util.request_attempt(proxies, mobile_num, res_handler, threadId)
            if ret:
                break
            else:
                if len(updated_proxies) != 0:
                    proxies = updated_proxies



def main():
    thread_num = 1
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

if __name__ == "__main__":
    main()