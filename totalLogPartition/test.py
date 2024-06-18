import re, urllib.parse, base64


res = urllib.parse.quote('')
print(res)

fwrite_flag = urllib.parse.quote(
                base64.b64encode(
                    '房间_房/间已满或重复进入-+添加失败'.encode('utf-8')
                ).decode('utf-8'),
                safe=''
            )
print(fwrite_flag)


# file_path = 'D:/Svnrepos/开发组文档/数据/projectx-docs/集群技术文档/workflow_backup/20210223/workflow.json'

# fw = open('test.log', mode='a', encoding='utf-8')
# fw_except = open('test_except.log', mode='a', encoding='utf-8')

# with open(file_path, mode='r', encoding='utf-8') as f:
#     for line in f:
#         # print(line)
#         if re.match(u'.*module_name=', line):
#             res = re.findall(u'.*(module_name.*?=.*?[\'"]{1}.+?[\'"]{1}[\s\S]*?[aAnNdD]*[\s\S]*?module_comment.*?=.*?[\'"]{1}.+?[\'"]{1})', line)
#             if not res:
#                 res = re.findall(u'.*(module_name.*?=.*?[\'"]{1}.+?[\'"]{1}.+?and.+?module_comment.*?in.*?\(.+?\))', line)
#                 # print(res, line)
#
#             if not res:
#                 fw_except.write(line + '\n')
#             # fw.write(line)
#             fw.write(str(res) + '\n')


# if re.search(u'^[^\[\]{}:。？！，、；：“”‘`（）《》〈〉【】『』「」﹃﹄〔〕…—～﹏￥]+$', 'abc1_-，23'):
#     print('matched')

# if not re.search(u'^(0x[0-9a-f]+|[0-9]+)$', '0213123asdf'):
#     print('123')
# if re.match(u'[a-zA-Z]+', 'abc521_34'):
#     print('matched')