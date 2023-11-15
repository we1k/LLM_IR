import json 
import torch
from tqdm import tqdm
import random
import argparse
# from langchain.llms import ChatGLM
from chatglm3.ChatGLM import ChatGLM
from typing import List, Optional
from utils import clean_question,clean_related_str,seed_everything

endpoint_url = ("http://127.0.0.1:29501")

config = {
        "temperature": 0.05,
        "top_p": 0.6,
        "threshold" : -40,
        "max_token" : 4096,
        # "with_history": True,
    }

def get_answer(data,prompt_template,abbre_dict,
               loop = True) -> str:
    """
    通过 ChatGLM 进行QA
    """

    system_info = {"role": "system", "content": "你是一位智能汽车说明的问答助手，你将根据节选的说明书的信息，完整地回答问题。"}
    chatglm = ChatGLM(
        endpoint_url=endpoint_url,
        history=[system_info],
        **config,
        model_kwargs={"sample_model_args": False},
    )

    inputs = {
        "question": clean_question(data["question"],abbre_dict),
        "info":clean_related_str(data["question"],data["related_str"],data["keyword"])
        }

    # Execute the chain
    user_info = ""
    for i in range(len(inputs["info"])):
        user_info += "第{}条相关信息：\n{}\n".format(i+1,inputs["info"][i]) 
    response0 = chatglm(prompt_template[0].format(inputs["question"],user_info))
    print(prompt_template[0].format(inputs["question"],user_info))
    response = ""
    if loop:
        response = chatglm(prompt_template[1].format(inputs["question"],response0))

    return [response0,response]

def write_json(results,output_path):
    # 读取JSON文件
    input_file = 'result/best_76.75.json'

    with open(input_file, 'r', encoding="utf-8") as file:
        data = json.load(file)

    # 保留question和answer字段
    filtered_data = []
    for item,result in zip(data,results):
        filtered_item = {
            "question": result.get("question"),
            "answer_1": result.get("answer_1").replace("\n", ""),
            "answer_2": result.get("answer_2").replace("\n", ""),
            "answer_3": item.get("answer_3").replace("\n", "")
        }
        filtered_data.append(filtered_item)

    # 写入新的JSON文件
    with open(output_path, 'w', encoding="utf-8") as file:
        json.dump(filtered_data, file,ensure_ascii=False,indent=4)

def main(opt):
    input_file = 'translate.json'
    with open(input_file, 'r', encoding='utf-8') as file:
        abbre_dict = json.load(file)

    if opt.test:
        data_path = "result/related_str_test.json"
    else:
        data_path = "result/related_str.json"
    with open(data_path, "r", encoding="utf-8") as file:
        json_data = file.read()
    datas = json.loads(json_data)

    prompt_template = ["""请根据说明书中提取的已知信息，简洁准确地回答问题。注意，相关信息的顺序不决定它的重要性，问题可能出现错别字，例如反光境是反光镜。问题是：{} 已知信息是：{} 答案是：""",
        """请根据问题尽可能简要地总结先前的回答，去掉与问题不相关的部分。问题是：{} \n先前的回答：\n{} 答案是："""]

    seed_everything(2023)
    results = []
    for data in tqdm(datas,desc="question"):
        ret = get_answer(data,prompt_template,abbre_dict,loop=False)
        sample = {"question": data["question"], "answer_1": ret[0], "answer_2": ret[1]}
        results.append(sample)
        print(sample["answer_1"])

    if opt.test == False :
        write_json(results=results,output_path="result/" + opt.output + "{}.json".format(opt.k))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action="store_true", help="whether to test")
    parser.add_argument('--output', type=str, default='final')
    opt = parser.parse_args()
    main(opt)

