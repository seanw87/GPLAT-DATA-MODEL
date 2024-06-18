import re

print(len(" 双冠模式   组队 ( 1 / 2 )   寻找 一起 合作 的 伙伴 ~   1669583190   85da5e20b150f6b8dd563a3614e4eb8a   20 团战 模式   组队 ( 1 / 5 )   寻找 一起 合作 的 伙伴 ~   1669583190   ed867b2e8109667a7c7c2f977b51b1c8   1 双冠模式   组队 ( 1 / 2 )   寻找 一起 合作 的 伙伴 ~   1669583190   d317fd00ce26413bd2690ae4051cd4f7   20 退 了 怎么 啦 挑战赛 :   双冠   组队 ( 1 / 2 )                 胜场 ： 1   寻找 一起 合作 的 伙伴 ~   1669583190   dddb1fee4cd9742f6288c43b206ff0cb   31".replace(" ", "")))


print(len("te st".replace(" ", "")))


itm = "双冠模式 组队(1/2) 会玩来 1440710705 8efd5cd7ce41c0429eafa24e75c3d919 20"
res = re.search('^.* 组队\([0-9]+/[0-9]+\) .*$', itm)
print(res)

print("select * from qqdz_structured.total_log_orc where dates='${days1ago}'".replace(' qqdz_structured.total_log_orc', ' transwarp.qqdz_structured.total_log_orc'))


chatresult = [['a', 1], ['a', 0], ['a', 0], ['a', 1]]
result_list = [0, 1, 1, 1]
for i, item in enumerate(chatresult):
    keyws = chatresult[i].pop()
    print('+++', keyws)
    chatresult[i].append(result_list[i])
    print('---', chatresult)
    chatresult[i].append(keyws)
print(chatresult)