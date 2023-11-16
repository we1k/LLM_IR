import json

# 读取JSON文件
input_glm = 'result/chatglm.json' # 保留answer_1
input_bc = 'result/baichuan.json' # 保留answer_1
input_qw = 'result/qianwen.json' # 保留answer_1
output_file = 'result/submit.json'

with open(input_glm, 'r', encoding="utf-8") as file:
    data1 = json.load(file)

with open(input_bc, 'r', encoding="utf-8") as file:
    data2 = json.load(file)

with open(input_qw, 'r', encoding="utf-8") as file:
    data3 = json.load(file)

# 保留question和answer字段
filtered_data = []
for item1,item2,item3 in zip(data1,data2,data3):
    filtered_item = {
        "question": item1.get("question"),
        "answer_1": item1.get("answer_1").replace("\n", ""),
        "answer_2": item2.get("answer_1").replace("\n", ""),
        "answer_3": item3.get("answer_1").replace("\n", "")
    }
    filtered_data.append(filtered_item)

# 写入新的JSON文件
with open(output_file, 'w', encoding="utf-8") as file:
    json.dump(filtered_data, file,ensure_ascii=False,indent=4)