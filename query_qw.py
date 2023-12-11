from transformers import AutoTokenizer, AutoModelForCausalLM
import json 
import torch
import argparse
from typing import List
from tqdm import tqdm, trange

from src.utils import clean_question,clean_related_str,seed_everything,write_json

def make_context(
    tokenizer,
    query: str,
    system: str = "",
    max_window_size: int = 6144,
    chat_format: str = "chatml",
):
    if chat_format == "chatml":
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [tokenizer.im_start_id]
        im_end_tokens = [tokenizer.im_end_id]
        nl_tokens = tokenizer.encode("\n")

        def _tokenize_str(role, content):
            return f"{role}\n{content}", tokenizer.encode(
                role, allowed_special=set()
            ) + nl_tokens + tokenizer.encode(content, allowed_special=set())

        system_text, system_tokens_part = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

        raw_text = ""
        context_tokens = []

        context_tokens = system_tokens + context_tokens
        raw_text = f"{im_start}{system_text}{im_end}" + raw_text
        context_tokens += (
            nl_tokens
            + im_start_tokens
            + _tokenize_str("user", query)[1]
            + im_end_tokens
            + nl_tokens
            + im_start_tokens
            + tokenizer.encode("assistant")
            + nl_tokens
        )
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

    elif chat_format == "raw":
        raw_text = query
        context_tokens = tokenizer.encode(raw_text)
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")

    return raw_text, context_tokens

def _decode_chatml(
    tokens: List[int],
    *,
    stop_words: List[str],
    eod_token_ids: List[int],
    tokenizer,
    raw_text_len: int,
    context_length: int,
    verbose: bool = False,
    return_end_reason: bool = False,
    errors: str='replace'
):
    end_reason = f"Gen length {len(tokens)}"
    eod_token_idx = context_length
    for eod_token_idx in range(context_length, len(tokens)):
        if tokens[eod_token_idx] in eod_token_ids:
            end_reason = f"Gen {tokenizer.decode([tokens[eod_token_idx]])!r}"
            break

    trim_decode_tokens = tokenizer.decode(tokens[:eod_token_idx], errors=errors)[raw_text_len:]
    if verbose:
        print("\nRaw Generate w/o EOD:", tokenizer.decode(tokens, errors=errors)[raw_text_len:])
        print("\nRaw Generate:", trim_decode_tokens)
        print("\nEnd Reason:", end_reason)
    for stop_word in stop_words:
        trim_decode_tokens = trim_decode_tokens.replace(stop_word, "").strip()
    trim_decode_tokens = trim_decode_tokens.strip()
    if verbose:
        print("\nGenerate:", trim_decode_tokens)

    if return_end_reason:
        return trim_decode_tokens, end_reason
    else:
        return trim_decode_tokens

def decode_tokens(
    tokens,
    tokenizer,
    raw_text_len: int,
    context_length: int,
    chat_format: str,
    verbose: bool = False,
    return_end_reason: bool = False,
    errors: str="replace",
) -> str:
    if torch.is_tensor(tokens):
        tokens = tokens.cpu().numpy().tolist()

    if chat_format == "chatml":
        return _decode_chatml(
            tokens,
            stop_words=[],
            eod_token_ids=[tokenizer.im_start_id, tokenizer.im_end_id],
            tokenizer=tokenizer,
            raw_text_len=raw_text_len,
            context_length=context_length,
            verbose=verbose,
            return_end_reason=return_end_reason,
            errors=errors,
        )
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")
    
def get_answer(datas,prompt_template,model,tokenizer,params) -> str:
    """
    通过 QianWen 进行QA, 同时使用batch_generate加速回答
    """
    all_raw_text = []
    for data in datas:
        inputs = {
            "question": data["question"],
            "info":data["related_str"]
            }
        
        user_info = ""
        for i in range(len(inputs["info"])):
            if len(user_info) + len(inputs["info"][i]) < params["max_length"]:
                user_info += "第{}条相关信息：\n{}\n".format(i+1,inputs["info"][i]) 
            else:
                user_info += "第{}条相关信息：\n{}\n".format(i+1,inputs["info"][i]) 
                user_info = user_info[:params["max_length"]]
                break
        all_raw_text.append(prompt_template[0].format(inputs["question"],user_info))
    
    batch_raw_text = []
    for q in all_raw_text:
        raw_text, _ = make_context(
            tokenizer,
            q,
            system="你是一位智能汽车说明的问答助手，你将根据节选的说明书的信息，完整并简洁地回答问题。",
            max_window_size=3072,
            chat_format='chatml',
        )
        batch_raw_text.append(raw_text)
    
    batch_input_ids = tokenizer(batch_raw_text, padding='longest', truncation=True, max_length=params["max_length"])
    batch_input_ids = torch.LongTensor(batch_input_ids['input_ids']).to(model.device)
    batch_out_ids = model.generate(
        batch_input_ids,
        return_dict_in_generate=False,
        generation_config=model.generation_config
    )
    padding_lens = [batch_input_ids[i].eq(tokenizer.pad_token_id).sum().item() for i in range(batch_input_ids.size(0))]

    batch_response = [
        decode_tokens(
            batch_out_ids[i][padding_lens[i]:],
            tokenizer,
            raw_text_len=len(batch_raw_text[i]),
            context_length=(batch_input_ids[i].size(0)-padding_lens[i]),
            chat_format="chatml",
            verbose=False,
            errors='replace'
        ) for i in range(len(all_raw_text))
    ]
    
    return batch_response


def main(opt):
    device = 'cuda:{}'.format(opt.device)

    model_name_or_path = "/tcdata/qwen/Qwen-7B-Chat"
    if opt.use_14B:
        model_name_or_path = "/tcdata/qwen/Qwen-14B-Chat-Int4"
    if opt.local_run:
        model_name_or_path = '.' + model_name_or_path
        
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, 
                                              trust_remote_code=True,
                                              pad_token='<|extra_0|>',
                                              eos_token='<|endoftext|>',
                                              padding_side = 'left')
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, 
                                                   pad_token_id=tokenizer.pad_token_id,
                                                   trust_remote_code=True, device_map=device)
    model = model.eval()

    if opt.test:
        data_path = "result/related_str_test.json"
    else:
        data_path = "result/related_str.json"
    print(data_path)
    with open(data_path, "r", encoding="utf-8") as file:
        json_data = file.read()
    datas = json.loads(json_data)

    prompt_template = ["""你是一位智能汽车使用说明的问答助手，现在需要从已有信息保留与问题最相关的部分，简要地回答问题，回答问题时最好保留原文的风格。问题是：{} \n{}答案是：""",
        """请尽可能简要地总结先前的回答，只保留与问题最相关的部分，在总结中不要重复问题。问题是：{} 答案是："""]
    
    max_length = 2048
    top_p       = opt.top_p
    temperature = opt.temperature
    params = {"max_length":max_length,"top_p":top_p,"temperature":temperature}

    seed_everything(opt.seed)
    batch_size = 2
    results = []
    for i in trange(0,len(datas),batch_size,desc="question"):
        if i+batch_size > len(datas):
            data = datas[i:]
        else:
            data = datas[i:i+batch_size]

        ret = get_answer(data,prompt_template,model,tokenizer,params)
        for j in range(len(data)):
            sample = {"question": data[j]["question"], "answer_1": ret[j]}
            results.append(sample)


    if opt.test == False :
        write_json(results=results,output_path="result/" + opt.output + ".json")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action="store_true", help="whether to test")
    parser.add_argument('--output', type=str, default='qianwen')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument("--temperature", default=0.5, type=float)
    parser.add_argument("--top_p", default=0.6, type=float)
    parser.add_argument("--local_run", action="store_true")
    parser.add_argument("--use-14B", action="store_true")
    opt = parser.parse_args()
    main(opt)
