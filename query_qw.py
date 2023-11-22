from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig 
import json 
import torch
from tqdm import trange
from typing import List
import argparse

from src.utils import clean_question,clean_related_str,seed_everything,write_json

def make_context(
    tokenizer,
    query: str,
    history,
    system: str = "",
    max_window_size: int = 6144,
    chat_format: str = "chatml",
):
    if history is None:
        history = []

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
                break

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

def get_answer(datas,prompt_template,model,tokenizer,params,abbre_dict) -> str:
    """
    通过 QianWen 进行QA, 同时使用batch_generate加速回答
    """
    all_raw_text = []
    for data in datas:
        inputs = {
            "question": clean_question(data["question"],abbre_dict),
            "info":clean_related_str(data["question"],data["related_str"],data["keyword"])
            }
        
        user_info = ""
        for i in range(len(inputs["info"])):
            user_info += "第{}条相关信息：\n{}\n".format(i+1,inputs["info"][i]) 
        all_raw_text.append(prompt_template[0].format(inputs["question"],user_info))
    
    batch_raw_text = []
    for q in all_raw_text:
        raw_text, _ = make_context(
            tokenizer,
            q,
            system="你是一位智能汽车说明的问答助手，你将根据节选的说明书的信息，完整并简洁地回答问题。",
            max_window_size=model.generation_config.max_window_size,
            chat_format=model.generation_config.chat_format,
        )
        batch_raw_text.append(raw_text)
    
    batch_input_ids = tokenizer(batch_raw_text, padding='longest')
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
    if opt.local_run:
        chatglm_path = "/home/lzw/.hf_models/Qwen-7B-Chat"
    else:
        chatglm_path = "/app/models/Qwen-7B-Chat"
    tokenizer = AutoTokenizer.from_pretrained(chatglm_path, 
                                              trust_remote_code=True,
                                              pad_token='<|extra_0|>',
                                              eos_token='<|endoftext|>',
                                              padding_side = 'left')
    model = AutoModelForCausalLM.from_pretrained(chatglm_path, 
                                                   pad_token_id=tokenizer.pad_token_id,
                                                   trust_remote_code=True, device_map=device).eval()
    max_new_tokens = 4096
    top_p       = opt.top_p
    temperature = opt.temperature
    params = {"max_new_tokens":max_new_tokens,"top_p":top_p,"temperature":temperature}

    model.generation_config = GenerationConfig.from_pretrained(chatglm_path, pad_token_id=tokenizer.pad_token_id,
                                                               **params)

    input_file = 'data/abbr2word.json'
    with open(input_file, 'r', encoding='utf-8') as file:
        abbre_dict = json.load(file)
    if opt.test:
        data_path = "result/related_str_test.json"
    else:
        data_path = "result/related_str.json"
    print(data_path)
    with open(data_path, "r", encoding="utf-8") as file:
        json_data = file.read()
    datas = json.loads(json_data)

    prompt_template = ["""你是一位智能汽车使用说明的问答助手，现在需要从已有信息保留与问题最相关的部分，简要地回答问题，回答问题时最好保留原文的风格。问题是：{} \n{}答案是：""",]
    
    seed_everything(2023)
    batch_size = 2
    results = []
    for i in trange(0,len(datas),batch_size,desc="question"):
        ret = get_answer(datas[i:i+batch_size],prompt_template,model,tokenizer,params,abbre_dict)
        for j in range(batch_size):
            sample = {"question": datas[i+j]["question"], "answer_1": ret[j]}
            results.append(sample)
            # print(sample)

    if opt.test == False :
        write_json(results=results,output_path="result/" + opt.output + ".json")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action="store_true", help="whether to test")
    parser.add_argument('--output', type=str, default='qianwen')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--temperature", default=0.5, type=float)
    parser.add_argument("--top_p", default=0.6, type=float)
    parser.add_argument("--local_run", action="store_true")
    opt = parser.parse_args()
    main(opt)
