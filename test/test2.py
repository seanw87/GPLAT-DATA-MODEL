print(str(141123)[0:3])

dict1 = {"a":1, "b":2}
dict2 = {"a":2, "c":3}

temp = {**dict1, **dict2}
print(temp)

list1 = [1,2,3]
list2 = [3,4,5]
print(list1 + list2)


a = [t for t in list1 if t > 1]
print(a)

var1='he'
var2='bastard'
print("{} is a {}".format(var1, var2));


import pandas as pd

activity_missed_items_withtype = [
    (1, 2, 0.009),
    (2, 2, 0.007),
    (3, 3, 0.004),
    (4, 3, 0.003),
    (5, 3, 0.002),
    (6, 3, 0.001),
    (7, 3, 0.09),
    (8, 3, 0.08),
    (9, 3, 0.07),
    (10, 3, 0.06),
    (11, 3, 0.05),
    (12, 3, 0.03),
]
item_activity_asso = {1: 123,2: 321, 3: 123, 4: 321, 5: 123, 6: 321, 7: 123, 8: 321, 9: 123, 10: 321, 11: 123, 12: 321}

activity_df = pd.DataFrame({
    'itemid': [t[0] for t in activity_missed_items_withtype],
    'itemtype': [t[1] for t in activity_missed_items_withtype],
    'activity_id': [item_activity_asso[t[0]] for t in activity_missed_items_withtype],
    'score': [t[2] for t in activity_missed_items_withtype]
})
activity_df.sort_values(['activity_id', 'score'], ascending=[1, 0], inplace=True)
activity_grouped = activity_df.groupby(['activity_id']).head(3)
activity_grouped = activity_grouped.drop(columns=['activity_id'])

activity_missing_items_withtype = activity_grouped.values.tolist()
print(activity_missing_items_withtype)



import re
res = re.sub(r'[^\u4e00-\u9fa5_0-9a-zA-Z]', "_", "拉请稍等控件放_(*&^adskfj091293_")
print(res)
