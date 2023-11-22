import os
import json

# 读取JSON文件
input_glm = 'result/chatglm.json' # 保留answer_1
input_bc = 'result/baichuan.json' # 保留answer_1
input_qw = 'result/qianwen.json' # 保留answer_1
output_file = 'result/submit.json'

datas = []
if os.path.exists(input_glm):
    with open(input_glm, 'r', encoding="utf-8") as file:
        datas.append(json.load(file))

if os.path.exists(input_bc):
    with open(input_bc, 'r', encoding="utf-8") as file:
        datas.append(json.load(file))

if os.path.exists(input_qw):
    with open(input_qw, 'r', encoding="utf-8") as file:
        datas.append(json.load(file))

# 保留question和answer字段
filtered_data = []
for items in zip(*datas):
    filtered_item = {
        "question": items[0].get("question"),
        "answer_1": "",
        "answer_2": "",
        "answer_3": "",
    }
    # Assuming the question is the same in all files
    for i, item in enumerate(items):
        filtered_item[f"answer_{i+1}"] = item.get("answer_1", "").replace("\n", "")
    filtered_data.append(filtered_item)

# 写入新的JSON文件
with open(output_file, 'w', encoding="utf-8") as file:
    json.dump(filtered_data, file,ensure_ascii=False,indent=4)