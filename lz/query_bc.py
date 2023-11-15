from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
import json 
import torch
from tqdm import tqdm
import random
import argparse

from utils import clean_question,clean_related_str,seed_everything,get_answer,write_json

def main(opt):
    device = 'cuda:' + str(opt.device)
    input_file = 'translate.json'
    with open(input_file, 'r', encoding='utf-8') as file:
        abbre_dict = json.load(file)

    chatglm_path = "baichuan-inc/Baichuan2-7B-Chat"
    tokenizer = AutoTokenizer.from_pretrained(chatglm_path, use_fast=False, trust_remote_code=True)
    chatglm = AutoModelForCausalLM.from_pretrained(chatglm_path,device_map=device, torch_dtype=torch.bfloat16, trust_remote_code=True)
    chatglm.generation_config = GenerationConfig.from_pretrained(chatglm_path)
    chatglm = chatglm.eval()
    if opt.test:
        data_path = "result/related_str_test.json"
    else:
        data_path = "result/threshold-120_temperature0.5_top_p0.6_answer.json"
    with open(data_path, "r", encoding="utf-8") as file:
        json_data = file.read()
    datas = json.loads(json_data)

    prompt_template = ["""请根据说明书中提取的已知信息，完整简要地回答问题。注意，问题可能出现错别字，例如反光境是反光镜。问题是：{}\n{} 答案是：""",
        """请尽可能简要地总结先前的回答，只保留与问题最相关的部分，在总结中不要重复问题。问题是：{} 答案是："""]

    seed_everything(2023)
    results = []
    for data in tqdm(datas,desc="question"):
        ret = get_answer(data,prompt_template,chatglm,tokenizer,abbre_dict,loop=False)
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
    parser.add_argument('--device', type=int, default=3)
    opt = parser.parse_args()
    main(opt)
