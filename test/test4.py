import ast
import time

conf_item_material_str = "{7000407:250,7000408:70,7000409:27}"
for conf_item_material_itmstr in  conf_item_material_str.split(","):
    conf_item_material = conf_item_material_itmstr.split(":")

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
for n in range(0, 500000):
    try:
        conf_item_material = ast.literal_eval(conf_item_material_str)
    except SyntaxError:
        print("SyntasError")
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )

for n in range(0, 500000):
    try:
        conf_item_material = eval(conf_item_material_str)
    except SyntaxError:
        print("SyntasError")
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )

print(conf_item_material[7000409])

