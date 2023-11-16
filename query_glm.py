from transformers import AutoTokenizer, AutoModel
import json 
from tqdm import tqdm
import argparse

from src.utils import clean_question,clean_related_str,seed_everything,check_string,remove_excess,write_json


def get_answer(data,prompt_template,chatglm,tokenizer,params,abbre_dict,
               loop = True) -> str:
    """
    通过 ChatGLM 进行QA
    """

    inputs = {
        "question": clean_question(data["question"],abbre_dict),
        "info":clean_related_str(data["question"],data["related_str"],data["keyword"])
        # "info": data["related_str"]
        }

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

def main(opt):
    input_file = 'data/abbr2word.json'
    with open(input_file, 'r', encoding='utf-8') as file:
        abbre_dict = json.load(file)
    

    device = 'cuda:0'
    if opt.local_run:
        chatglm_path = "/home/lzw/.hf_models/chatglm3-6b"
    else:
        chatglm_path = "/app/models/chatglm3-6b"

    tokenizer = AutoTokenizer.from_pretrained(chatglm_path, use_fast=False, trust_remote_code=True)
    chatglm = AutoModel.from_pretrained(chatglm_path,device_map=device, trust_remote_code=True)
    chatglm = chatglm.eval()


    if opt.test:
        data_path = "result/related_str_test.json"
    else:
        data_path = "result/related_str.json"  

    with open(data_path, "r", encoding="utf-8") as file:
        json_data = file.read()
    datas = json.loads(json_data)

    prompt_template = ["""你是一位智能汽车使用说明的问答助手，现在需要从已有信息保留与问题最相关的部分，完整并简要地回答问题。问题是：{} \n{}答案是：""",
        """请尽可能简要地总结先前的回答，只保留与问题最相关的部分，在总结中不要重复问题。问题是：{} 答案是："""]

    max_length = 4096
    top_p      = 0.6
    temperature= 0.5
    params = {"max_length":max_length,"top_p":top_p,"temperature":temperature}

    seed_everything(2023)
    results = []
    for data in tqdm(datas,desc="question"):
        if check_string(data["question"],data["related_str"]):
            ret = [remove_excess(data["related_str"][0]),""]
        else:
            ret = get_answer(data,prompt_template,chatglm,tokenizer,params,abbre_dict,loop=False)
        sample = {"question": data["question"], "answer_1": ret[0], "answer_2": ret[1]}
        results.append(sample)
        # print(sample["answer_1"])
        # print("---")
        # print(sample["answer_2"])

    if opt.test == False :
        write_json(results=results,output_path="result/" + opt.output + ".json")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action="store_true", help="whether to test")
    parser.add_argument('--output', type=str, default='chatglm')
    parser.add_argument("--local_run", action="store_true")
    opt = parser.parse_args()
    main(opt)

