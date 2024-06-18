import os, datetime, glob, re, sys, base64, urllib.parse, threading

MODULE_NAME_BLACKLIST = ['msgfilter', 'DEBUG', 'debug', 'msgshumeifilter', 'test', 'Debug']
MODULE_COMMENT_BLACKLIST = ['']
MODULE_PATTERN = u'^[^\[\]{}:。？！，、；：“”‘`（）《》〈〉【】『』「」﹃﹄〔〕…—～﹏￥]+$'

TOTAL_LOG_DIR='data/'
# TOTAL_LOG_DIR='/backup/'
TOTAL_LOG_PARTITION_DIR='data/target'
TOTAL_LOG_PARTITION_LOG_DIR='log'

WORKERS = 10

FH_LOG = None
FH_MODULE_TARGET = None
FH_MODULE_DIR_CREATION_ABNORMAL = None
FH_MODULE_UNMATCHED = None

FWRITEHANDLE = dict()
DEBUG_DICT = dict()

LOG_PARTITION_DATA = dict()


# 多线程注入
class FileHandlerThread(threading.Thread):

    def __init__(self, func, args):
        super(FileHandlerThread, self).__init__()
        self.args = args
        self.func = func
        self.result = None

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None




# 获取当前日志所在小时分区
def get_total_log_file_dt():
    curdatetime = datetime.datetime.now()
    onehour = datetime.timedelta(hours=1)
    onehourago = curdatetime - onehour
    logname_dt = onehourago.strftime('%Y-%m-%d-%H')
    hour = onehourago.strftime('%H')
    # logname_dt = '2021-02-23-10'
    # hour = '10'
    return logname_dt, hour

# 获取total_log文件路径列表
def get_total_log_file_list(fname_wildcard):
    total_log_path = TOTAL_LOG_DIR + fname_wildcard
    flist = glob.glob(total_log_path)
    return flist

# 创建日志总目录
def create_log_dir(logdir):
    if not os.path.exists(logdir):
        try:
            print('Notice, 日志目录：', logdir, ' 不存在，新建')
            os.mkdir(logdir)
        except:
            print('Error, 日志目录：', logdir, ' 创建失败, 详情：', sys.exc_info())
            print('正在退出，请确认!')
            return False
        else:
            print('Notice, 日志目录：', logdir, ' 创建成功')
    return True

# 通用日志函数
def log(content, fh, level = 'ERROR'):
    curdt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_msg = curdt + ', ' + level + ', ' + content + '\n'
    fh.write(log_msg)

# 打开日志文件句柄
def open_log_fh(path, mode='a', encoding='utf-8'):
    try:
        fh = open(path, mode=mode, encoding=encoding)
    except:
        print('Error, 日志文件: ', path, ' 句柄打开失败 ', sys.exc_info())
        exit(0)
    else:
        return fh

# 通过byte分割文件过程中删除多余的头部行片段
def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)

# ↑ 按行处理
def proc_linebyline(game_flag, hour, line):
    global LOG_PARTITION_DATA
    modules = line.split('|')
    if len(modules) >= 3:
        module_name = modules[1]
        module_comment = modules[2]
        module_pair = module_name + '_' + module_comment

        if re.search(MODULE_PATTERN, module_name) \
                and not re.search(u'^(0x[0-9a-f]+|[0-9]+)$', module_name) \
                and module_name not in MODULE_NAME_BLACKLIST \
                and re.match(MODULE_PATTERN, module_comment) \
                and not re.search(u'^(0x[0-9a-f]+|[0-9]+)$', module_comment) \
                and module_comment not in MODULE_COMMENT_BLACKLIST:

            # fwrite_flag = urllib.parse.quote(
            #     base64.b64encode(
            #         module_pair.encode('utf-8')
            #     ).decode('utf-8'),
            #     safe=''
            # )

            # if fwrite_flag not in FWRITEHANDLE:
            #     target_dir = TOTAL_LOG_PARTITION_DIR + '/gflag=' + game_flag + '/mod=' + fwrite_flag
            #     try:
            #         os.makedirs(target_dir, exist_ok=True)
            #         FWRITEHANDLE[fwrite_flag] = open(target_dir + '/' + hour + '.log', mode='a',
            #                                          encoding='utf-8')
            #     except:
            #         print('Error, 目录: ', target_dir + '/' + hour + '.log', sys.exc_info())
            #         FH_MODULE_DIR_CREATION_ABNORMAL.write(target_dir + '\n')                    # for debug
            #     else:
            #         FH_MODULE_TARGET.write(module_pair + '\n')                                  # for debug
            #         FWRITEHANDLE[fwrite_flag].write(line)
            # else:
            #     FWRITEHANDLE[fwrite_flag].write(line)

            if module_pair not in LOG_PARTITION_DATA:
                LOG_PARTITION_DATA[module_pair] = list()
            LOG_PARTITION_DATA[module_pair].append(line)
        else:
            if module_pair not in DEBUG_DICT:                                                   # for debug
                DEBUG_DICT[module_pair] = 1                                                     # for debug
                FH_MODULE_UNMATCHED.write(module_pair + '\n')                                   # for debug

# ↑↑ 拆分为文件块处理
def proc_file_by_block(filename, hour, worker_id, num_workers):
    game_flag = filename.split('-')[0]
    with open(os.path.join(TOTAL_LOG_DIR, filename), encoding='UTF-8') as f:
        size = os.fstat(f.fileno()).st_size  # 指针操作，所以无视文件大小
        chunk_size = size // num_workers
        offset = worker_id * chunk_size
        end = offset + chunk_size
        f.seek(offset)
        print('worker_id: ', worker_id, ', size: ', size, ', chunk_size: ', chunk_size,
              ', offset: ', offset, ', end: ', end)
        if offset > 0:
            print(safe_readline(f))  # drop first incomplete line

        res = dict()

        line = f.readline()
        while line:
            if not line:
                line = f.readline()
                continue

            # proc_linebyline(game_flag, hour, line)
            modules = line.split('|')
            if len(modules) >= 3:
                module_name = modules[1]
                module_comment = modules[2]
                module_pair = module_name + '_' + module_comment

                if re.search(MODULE_PATTERN, module_name) \
                        and not re.search(u'^(0x[0-9a-f]+|[0-9]+)$', module_name) \
                        and module_name not in MODULE_NAME_BLACKLIST \
                        and re.match(MODULE_PATTERN, module_comment) \
                        and not re.search(u'^(0x[0-9a-f]+|[0-9]+)$', module_comment) \
                        and module_comment not in MODULE_COMMENT_BLACKLIST:

                    if module_pair not in res:
                        res[module_pair] = list()
                    res[module_pair].append(line)
                # else:
                #     if module_pair not in DEBUG_DICT:  # for debug
                #         DEBUG_DICT[module_pair] = 1  # for debug
                #         FH_MODULE_UNMATCHED.write(module_pair + '\n')  # for debug

            if f.tell() > end:
                break
            line = f.readline()
        return res

# ↑↑↑ 处理文件列表
def proc_filelist(flist, hour):
    # global FWRITEHANDLE
    # global DEBUG_DICT
    # global FH_MODULE_TARGET
    # global FH_MODULE_DIR_CREATION_ABNORMAL
    # global FH_MODULE_UNMATCHED

    # global LOG_PARTITION_DATA

    # 遍历total_log文件
    for filepath in flist:
        # FWRITEHANDLE = dict()
        # DEBUG_DICT = dict()
        # LOG_PARTITION_DATA = dict()

        filepath_list = os.path.split(filepath)
        file = filepath_list[len(filepath_list) - 1]

        game_flag = file.split('-')[0]
        print(file, '-', game_flag)

        # FH_MODULE_TARGET = open_log_fh(os.path.join(log_dir, game_flag + '-module_target.log'))
        # FH_MODULE_DIR_CREATION_ABNORMAL = open_log_fh(
        #     os.path.join(log_dir, game_flag + '-module_dir_creation_abnormal.log'))
        # FH_MODULE_UNMATCHED = open_log_fh(os.path.join(log_dir, game_flag + '-module_unmatched.log'))

        filename = game_flag + '-' + logname_dt + '.log'
        # print(glob.glob(TOTAL_LOG_DIR + '/' + filename))

        workers = WORKERS       # todo...根据不同文件大小设置不同并发度
        workers_thread = []
        for worker_id in range(workers):
            w = FileHandlerThread(proc_file_by_block, args=(filename, hour, worker_id, workers))
            workers_thread.append(w)
            w.start()
        for w in workers_thread:
            w.join()
        for w in workers_thread:
            res = w.get_result()
            for module_pair in res:
                fwrite_flag = urllib.parse.quote(
                    base64.b64encode(
                        module_pair.encode('utf-8')
                    ).decode('utf-8'),
                    safe=''
                )
                target_dir = TOTAL_LOG_PARTITION_DIR + '/gflag=' + game_flag + '/mod=' + fwrite_flag
                try:
                    os.makedirs(target_dir, exist_ok=True)
                    fh = open(target_dir + '/' + hour + '.log', mode='a',
                                                     encoding='utf-8')
                except:
                    print('Error, 目录: ', target_dir + '/' + hour + '.log', sys.exc_info())
                    FH_MODULE_DIR_CREATION_ABNORMAL.write(target_dir + '\n')                    # for debug
                else:
                    # FH_MODULE_TARGET.write(module_pair + '\n')                                  # for debug
                    for line in res[module_pair]:
                        fh.write(line)
                    fh.close()





if __name__ == '__main__':
    print('-'*10 + ' starting... ' + '-'*10)
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    logname_dt, hour = get_total_log_file_dt()

    # 创建日志目录
    if not create_log_dir(TOTAL_LOG_PARTITION_LOG_DIR):
        exit(0)
    log_dir = os.path.join(TOTAL_LOG_PARTITION_LOG_DIR, logname_dt)
    if not create_log_dir(log_dir):
        exit(0)

    FH_LOG = open_log_fh(os.path.join(TOTAL_LOG_PARTITION_LOG_DIR, 'main.log'))

    # fname_wildcard = '[!1-2]-' + logname_dt + '.log'
    fname_wildcard = '[1-2]-' + logname_dt + '.log'
    flist = get_total_log_file_list(fname_wildcard)

    # 文件依次处理
    proc_filelist(flist, hour)

    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    print('-' * 10 + ' fin ' + '-' * 10)








        # with open(os.path.join(TOTAL_LOG_DIR, filename), encoding='UTF-8') as f:
        #     for idx, line in tqdm(enumerate(f)):
        #         modules = line.split('|')
        #         if len(modules)>=3:
        #             module_name = modules[1]
        #             module_comment = modules[2]
        #             module_pair = module_name + '_' + module_comment

        #             if re.search(MODULE_PATTERN, module_name)\
        #                 and not re.search(u'^(0x[0-9a-f]+|[0-9]+)$', module_name)\
        #                 and module_name not in MODULE_NAME_BLACKLIST\
        #                 and re.match(MODULE_PATTERN, module_comment)\
        #                 and not re.search(u'^(0x[0-9a-f]+|[0-9]+)$', module_comment)\
        #                 and module_comment not in MODULE_COMMENT_BLACKLIST:
        #
        #                 fwrite_flag = urllib.parse.quote(
        #                     base64.b64encode(
        #                         module_pair.encode('utf-8')
        #                     ).decode('utf-8')
        #                 )
        #
        #                 if fwrite_flag not in FWRITEHANDLE:
        #                     target_dir = TOTAL_LOG_PARTITION_DIR + '/gflag=' + game_flag + '/mod=' + fwrite_flag
        #                     try:
        #                         os.makedirs(target_dir, exist_ok=True)
        #                         FWRITEHANDLE[fwrite_flag] = open(target_dir + '/' + game_flag + '.log', mode='a', encoding='utf-8')
        #                     except:
        #                         print('Error, 目录: ', target_dir + '/' + game_flag + '.log', sys.exc_info())
        #                         FH_MODULE_DIR_CREATION_ABNORMAL.write('/gflag=' + game_flag + '/mod=' + fwrite_flag + '\n')
        #                     else:
        #                         FH_MODULE_TARGET.write(module_pair + '\n')
        #                         FWRITEHANDLE[fwrite_flag].write(line)
        #                 else:
        #                     FWRITEHANDLE[fwrite_flag].write(line)
        #             else:
        #                 if module_pair not in DEBUG_DICT:
        #                     DEBUG_DICT[module_pair]=1
        #                     FH_MODULE_UNMATCHED.write(game_flag + '_' + module_pair + '\n')
