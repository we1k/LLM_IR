import json 
import torch
from tqdm import tqdm
import random
import argparse
# from langchain.llms import ChatGLM
from chatglm3.ChatGLM import ChatGLM
from typing import List, Optional

endpoint_url = ("http://127.0.0.1:29501")

config = {
        "temperature": 0.05,
        "top_p": 0.6,
        "threshold" : -40,
        "max_token" : 4096,
        # "with_history": True,
    }

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def clean_related_str(related_str:List, keyword: Optional[List]=[], threshold=-80):
    """
        Input:
            related_str: List[str]
        Return:
            List[str]
    """
    if len(related_str) == 1:
        return related_str

    ## remove previous section (if it contains the keyword)
    if len(keyword) > 0 and keyword[0][1] > threshold:
        potential_section = keyword[0][0] + "\n"
        for i in range(len(related_str)):
            if potential_section in related_str[i]:
                related_str[i] = potential_section + potential_section.join(related_str[i].split(potential_section)[1:])
            
    ## remove duplicate
    tmp_str = "\n".join(related_str)
    parts = tmp_str.split("\n")
    # print(parts)
    # print("--------------------------------")
    seen = set()
    deduplicated_parts = []
    # 去重，同时保证去重后的顺序不变
    for part in parts:
        if part not in seen:
            seen.add(part)
            deduplicated_parts.append(part)

    related_str = "\n".join(deduplicated_parts)
    # if len(related_str) > 1:
    #     print(related_str)
    return related_str

def get_answer(data,prompt_template,
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
        "question": data["question"],
        "info":clean_related_str(data["related_str"], data["keyword"])
        }

    # Execute the chain
    user_info = inputs["info"]
    # for i in range(len(inputs["info"])):
    #     user_info += "第{}条相关信息：\n{}\n".format(i+1,inputs["info"][i]) 

    response0 = chatglm(prompt_template[0].format(inputs["question"],user_info))
    # print(prompt_template[0].format(inputs["question"],user_info))
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
        ret = get_answer(data,prompt_template)
        sample = {"question": data["question"], "answer_1": ret[0], "answer_2": ret[1]}
        results.append(sample)
        print(sample["answer_1"])
        print("---")
        print(sample["answer_2"])

    write_json(results=results,output_path="result/" + opt.output + ".json")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action="store_true", help="whether to test")
    parser.add_argument('--output', type=str, default='final')
    opt = parser.parse_args()
    main(opt)

