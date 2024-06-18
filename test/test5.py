# -*- coding: utf-8 -*-
# tmp1 = [1,2,3,4,5,6,7]
# tmp2 = [10, 11, 12, 13, 14, 15]
#
# for t1 in tmp1:
#     for t2 in tmp2:
#         print(t1, t2)
#         if t2 > 12:
#             break
#
# print(len(tmp1))
#
# tmp1.append(10)
# print(tmp1)
#
# def test(para1, para2 = None):
#     if para2 is None:
#         print(2, para1)
#
#     print(1, para1)
#
#
# test("asdf", 111)
#
# import os
#
# print(os.path.join("a/b/c", "asbc"))
#
#
# for i in range(1, 2):
#     print(i)



feedback = u"太空号:20669724,我是船员，我有打陨石的任务，让他们去看，然后那时候刚好挂机，接了个电话没做，然后他们就一直在骂我，骂的特别特别难听，我让他们道歉，但是他们就是不道歉，还一直在骂我，然后全部都说要举报我，他们骂我，我让他们道歉，这有错吗？然后除了骂我的那两个人，其他人都说要举报我，他们就骂的特别难听，让他们道歉他们就又骂骂的特难听，你们去可以去看一下回放，是不是他们全程都在骂我，我只是想让他们说句对不起！然后如果我没猜错的话，应该他们全部都举报我了吧。他们全部说要举报我，我就特别难受特别想哭连想死的心都有了，不要觉得我特别矫情，被骂了谁心里还不难受？我希望他们被封号或者得到一个满意的答复好吗？谢谢！"
print(feedback)
# exit(0)
chars = list(feedback)
print(chars)
# exit(0)
#
char_freq = {}
for char in chars:
    if char not in char_freq:
        char_freq[char] = 1
    else:
        char_freq[char] += 1

for char in char_freq:
    print(char, char_freq[char])