import os
import json 
import torch
from tqdm import trange
from typing import List
import argparse

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig 

from src.prompt_template import PROMPT_TEMPLATE
from src.utils import clean_question,seed_everything,write_json

def get_prompt(datas, prompt_template, tokenizer,params, abbre_dict):
    all_raw_text = []
    for i, data in enumerate(datas):
        inputs = {
            "question": clean_question(data["question"],abbre_dict),
            "info":data["related_str"]
            }
        
        user_info = ""

        for j in range(len(inputs["info"])):
            if len(user_info) + len(inputs["info"][j]) < params["max_length"]:
                user_info += "第{}条相关信息：\n{}\n".format(j+1,inputs["info"][j]) 
            else:
                user_info += "第{}条相关信息：\n{}\n".format(j+1,inputs["info"][j]) 
                user_info = user_info[:params["max_length"]]
                break
        
        raw_text = make_context(
            tokenizer,
            prompt_template.format(inputs["question"],user_info),
            system="你是一位智能汽车说明的问答助手，你将根据节选的说明书的信息，完整并简洁地回答问题。",
            max_window_size=3072,# model.generation_config.max_window_size,
            chat_format="chatml", # model.generation_config.chat_format 
        )

        all_raw_text.append(raw_text)
    
    with open("result/prompt.json", "w", encoding="utf-8") as file:
        json.dump(all_raw_text, file, ensure_ascii=False, indent=4)

def make_context(
    tokenizer,
    query: str,
    system: str = "",
    max_window_size: int = 6144,
    chat_format: str = "chatml",
):
    history = []

    if chat_format == "chatml":
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [tokenizer.im_start_id]
        im_end_tokens = [tokenizer.im_end_id]
        nl_tokens = tokenizer.encode("\n")

        def _tokenize_str(role, content):
            return f"{role}\n{content}"

        system_text = _tokenize_str("system", system)
        system_tokens_part = tokenizer.encode(
                "system", allowed_special=set()
            ) + nl_tokens + tokenizer.encode(system, allowed_special=set())
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

        raw_text = ""
        context_tokens = []

        for turn_query, turn_response in reversed(history):
            query_text, query_tokens_part = _tokenize_str("user", turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
            response_text, response_tokens_part = _tokenize_str(
                "assistant", turn_response
            )
            response_tokens = im_start_tokens + response_tokens_part + im_end_tokens

            next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
            prev_chat = (
                f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
            )

            current_context_size = (
                len(system_tokens) + len(next_context_tokens) + len(context_tokens)
            )
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                context_tokens = (next_context_tokens + context_tokens)[:max_window_size]
                raw_text = (prev_chat + raw_text)[:max_window_size]
                break

        raw_text = f"{im_start}{system_text}{im_end}" + raw_text
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

    elif chat_format == "raw":
        raw_text = query
        context_tokens = tokenizer.encode(raw_text)
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")

    return raw_text

def get_answer(datas,prompt_template,model,tokenizer,params,abbre_dict,is_beam_search=False) -> str:
    all_raw_text = []
    all_text_ids = []
    output = ["无答案"] * len(datas)
    for i, data in enumerate(datas):
        inputs = {
            "question": clean_question(data["question"],abbre_dict),
            "info":data["related_str"]
            }
        
        user_info = ""
        # for j in range(len(inputs["info"])):
        #     user_info += "```{}```\n".format(j+1,inputs["info"][j]) 
        #     if len(user_info) >= params["max_length"]:
        #         user_info = user_info[:params["max_length"]]
        #         break
        for j in range(len(inputs["info"])):
            if len(user_info) + len(inputs["info"][j]) < params["max_length"]:
                user_info += "第{}条相关信息：\n{}\n".format(j+1,inputs["info"][j]) 
            else:
                user_info += "第{}条相关信息：\n{}\n".format(j+1,inputs["info"][j]) 
                user_info = user_info[:params["max_length"]]
                break
        
        # print(len(prompt_template))
        # if len(prompt_template) > 100:
        ####### few shot
        #     system_string = "你是一位智能汽车使用说明的问答助手，现在需要选取已知信息中与问题最相关的部分，完整并简要地回答问题。请参照问题1和问题2的示例进行回答。"
        # else:
        # system_string = "你是一位智能汽车使用说明的问答助手，现在需要选取已知信息中与问题最相关的部分，完整并简要地回答问题，回答问题时最好保留原文的风格。"
        
        raw_text = make_context(
            tokenizer,
            prompt_template.format(inputs["question"],user_info),
            system="你是一位智能汽车说明的问答助手，你将根据节选的说明书的信息，完整并简洁地回答问题。",
            max_window_size=3072,# model.generation_config.max_window_size,
            chat_format="chatml", # model.generation_config.chat_format 
        )

        # 无答案直接跳过
        if data['dont_answer'] == False:
            all_text_ids.append(i)
            all_raw_text.append(raw_text)
    

    params.pop("max_length")
    # print(params)
    sample_params = SamplingParams(**params,use_beam_search=is_beam_search, stop=["<|im_end|>"])
    preds = model.generate(all_raw_text, sample_params)
    # writing back
    for i, pred in enumerate(preds):
        output[all_text_ids[i]] = pred.outputs[0].text

    return output

def main(opt):
    seed_everything(opt.seed)

    model_name_or_path = "/tcdata/qwen/Qwen-7B-Chat"
    if opt.local_run:
        model_name_or_path = "./tcdata/qwen/Qwen-7B-Chat"
    if opt.use_14B:
        model_name_or_path = "./rerank_model/Qwen-14B-Chat-AWQ"
        if opt.local_run:
            model_name_or_path = "./tcdata/qwen/Qwen-14B-Chat-AWQ"
    if opt.use_1_8B:
        model_name_or_path = "./rerank_model/Qwen-1_8B-Chat"
        if opt.local_run:
            model_name_or_path = "./tcdata/qwen/Qwen-1_8B-Chat"

    print(model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, )
    max_tokens = 2048
    params = {"max_length":1024, "max_tokens":max_tokens,"top_p":opt.top_p,"temperature":opt.temperature}
    # generation_config = GenerationConfig.from_pretrained(model_name_or_path, pad_token_id=tokenizer.pad_token_id, **params, trust_remote_code=True)
    if opt.beam_search:
        params["temperature"] = 0
        params["top_p"] = 1
        params["best_of"] = opt.best_of
        params["n"] = 1
        params["length_penalty"] = opt.length_penalty

    # choose prompt
    prompt_template = PROMPT_TEMPLATE[opt.prompt_idx]

    input_file = 'data/abbr2word.json'
    with open(input_file, 'r', encoding='utf-8') as file:
        abbre_dict = json.load(file)
    if opt.test:
        data_path = "result/related_str_test.json"
    else:
        data_path = "result/related_str.json"
    
    with open(data_path, "r", encoding="utf-8") as file:
        json_data = file.read()
    datas = json.loads(json_data)

    max_model_len = 2048 if opt.use_14B else 4096

    if opt.local_run:
        llm = LLM(model=model_name_or_path, trust_remote_code=True, tensor_parallel_size=opt.tensor_parallel_size, gpu_memory_utilization=0.5, dtype=torch.float16, max_model_len=max_model_len)
    else:
        llm = LLM(model=model_name_or_path, trust_remote_code=True, tensor_parallel_size=opt.tensor_parallel_size, gpu_memory_utilization=0.97, dtype=torch.float16, max_model_len=max_model_len)

    outputs = get_answer(datas, prompt_template, llm, tokenizer, params, abbre_dict, opt.beam_search)

    # for output in outputs[:1]:
    #     prompt = output.prompt
    #     generated_text = output.outputs[0].text
    #     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    results = []
    for i, output in enumerate(outputs):
        results.append(
            {
                "question": datas[i]["question"],
                "answer_1": output 
            }
        )

    write_json(results=results,output_path="result/" + opt.output + ".json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action="store_true", help="whether to test")
    parser.add_argument('--output', type=str, default='qianwen')
    parser.add_argument('--prompt_idx', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument("--temperature", default=0.8, type=float)
    parser.add_argument("--top_p", default=0.9, type=float)
    parser.add_argument("--local_run", action="store_true")
    parser.add_argument("--use-14B", action="store_true")
    parser.add_argument("--use-1_8B", action="store_true")
    parser.add_argument("--beam_search", action="store_true")
    parser.add_argument("--best_of", type=int, default=3)
    parser.add_argument("--length_penalty", type=float, default=1)
    opt = parser.parse_args()
    main(opt)
