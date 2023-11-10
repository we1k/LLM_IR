from transformers import AutoTokenizer, AutoModel
import json 
import torch
from tqdm import tqdm
import random
import argparse
from fuzzywuzzy import fuzz,process

device = 'cuda:2'
chatglm_path = "chatglm3-6b/chatglm3-6b"
chatglm = AutoModel.from_pretrained(chatglm_path, trust_remote_code=True, device=device)
chatglm = chatglm.eval()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def adjusted_ratio_fn(str1, str2):
    # 根据字符串长度计算一个权重，用于调整相似度得分
    try:
        len_weight = min(len(str1), len(str2)) / max(len(str1), len(str2))
        
        # 计算标准的相似度得分
        similarity_score = fuzz.WRatio(str1, str2)
        
        # 根据长度权重调整得分
        score = similarity_score * len_weight
        return score
    except:
        print("len of string is zero")
        return 100

def clean_related_str(question,related_str, keyword, threshold=-80):
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
    tmp_str = "<BOS>\n".join(related_str)
    parts = tmp_str.split("\n")
    # print(parts)
    # print("--------------------------------")
    seen = set()
    deduplicated_parts = []
    # 去重，同时保证去重后的顺序不变
    for part in parts:
        nobospart = part.strip("。\n").replace("<SEP>","")
        if nobospart.startswith("<BOS>"):
            nobospart = nobospart[5:]
        if len(nobospart) == 0:
            continue
        if len(seen) > 0:
            _,max_score = process.extractOne(query=nobospart, choices=list(seen), scorer=adjusted_ratio_fn)
            if max_score > 95:
                # print("删除句子",nobospart)
                continue
        seen.add(nobospart)
        deduplicated_parts.append(part)

    tmp_str = "\n".join(deduplicated_parts)
    related_str = tmp_str.split("<BOS>")
    related_str = [str.strip("\n") for str in related_str]

    return related_str

def get_answer(data,prompt_template,chatglm,params,
               loop = True) -> str:
    """
    通过 ChatGLM 进行QA
    """

    inputs = {
        "question": data["question"],
        "info":clean_related_str(data["question"],data["related_str"],data["keyword"])
        # "info": data["related_str"]
        }

    tokenizer = AutoTokenizer.from_pretrained(chatglm_path, trust_remote_code=True)
    system_info = {"role": "system", "content": "你是一位智能汽车说明的问答助手，你将根据节选的说明书的信息，完整并简洁地回答问题。"}

    # Execute the chain
    user_info = ""
    for i in range(len(inputs["info"])):
        user_info += "第{}条相关信息：\n{}\n".format(i+1,inputs["info"][i]) 
    response0, history = chatglm.chat(tokenizer,prompt_template[0].format(inputs["question"],user_info), history=[system_info],**params)
    # print(prompt_template[0].format(inputs["question"],user_info))
    response = ""
    if loop:
        response, history = chatglm.chat(tokenizer,prompt_template[1].format(inputs["question"]), history=history,**params)

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
        if opt.k == 'more':
            data_path = "result/related_str.json"                 ## k更大
        else:
            data_path = "result/related/related_str1101-76.75.json" ## 前面一句后面两句 
    print(data_path)
    with open(data_path, "r", encoding="utf-8") as file:
        json_data = file.read()
    datas = json.loads(json_data)

    prompt_template = ["""你是一位智能汽车使用说明的问答助手，现在需要根据已有信息，完整简要地回答问题。问题是：{} \n{}答案是：""",
        """请尽可能简要地总结先前的回答，只保留与问题最相关的部分，在总结中不要重复问题。问题是：{} 答案是："""]

    max_length = 4096
    top_p      = 0.6
    temperature= 0.5
    params = {"max_length":max_length,"top_p":top_p,"temperature":temperature}

    seed_everything(2023)
    results = []
    for data in tqdm(datas,desc="question"):
        ret = get_answer(data,prompt_template,chatglm,params,loop=True)
        sample = {"question": data["question"], "answer_1": ret[0], "answer_2": ret[1]}
        results.append(sample)
        print(sample["answer_1"])
        print("---")
        print(sample["answer_2"])

    if opt.test == False :
        write_json(results=results,output_path="result/" + opt.output + "{}.json".format(opt.k))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action="store_true", help="whether to test")
    parser.add_argument('--output', type=str, default='final')
    parser.add_argument('--k', type=str, default='more', help="more or less")
    opt = parser.parse_args()
    main(opt)

