import json

# 读取JSON文件
input_file_1 = 'result/ret001.json' # 保留answer_2
input_file_2 = 'result/ret002.json' # 保留answer_2和answer_3
output_file = 'result/final21.json'

with open(input_file_1, 'r', encoding="utf-8") as file:
    data1 = json.load(file)

with open(input_file_2, 'r', encoding="utf-8") as file:
    data2 = json.load(file)

# 保留question和answer字段
filtered_data = []
for item1,item2 in zip(data1,data2):
    filtered_item = {
        "question": item1.get("question"),
        "answer_1": item1.get("answer_1").replace("\n", ""),
        "answer_2": item2.get("answer_1").replace("\n", ""),
        "answer_3": item2.get("answer_2").replace("\n", "")
    }
    filtered_data.append(filtered_item)

# 写入新的JSON文件
with open(output_file, 'w', encoding="utf-8") as file:
    json.dump(filtered_data, file,ensure_ascii=False,indent=4)