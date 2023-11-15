from fuzzywuzzy import fuzz,process
import torch
import random
import json

SHORTEST_LENGTH = 1

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

    # ## remove previous section (if it contains the keyword)
    # if len(keyword) > 0 and keyword[0][1] > threshold:
    #     potential_section = keyword[0][0] + "\n"
    #     for i in range(len(related_str)):
    #         if potential_section in related_str[i]:
    #             related_str[i] = potential_section + potential_section.join(related_str[i].split(potential_section)[1:])
            
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
        if len(nobospart) < SHORTEST_LENGTH:
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

def clean_question(question,abbre_dict):
    # 使用字典进行替换
    for abbreviation, translation in abbre_dict.items():
        question = question.replace(abbreviation, translation+'('+abbreviation+')')
    
    return question

def remove_excess(string):
    """
    对特殊问题，直接返回的related_str进行清洗
    """
    ## 清除小标题(section name)
    index = string.find("\n")
    if index != -1:
        string = string[index+1:]
    ## 清除后面的说明/警告
    keywords = ["说明", "警告"]
    for keyword in keywords:
        index = string.find(keyword)
        if index != -1:
            string = string[:index]
            break
    return string.strip()

def check_string(sentence,related_str):
    """
    检查是否属于特殊问题
    """
    ## 取出keyword，最相关的string的第一句
    parts = related_str[0].split("\n")
    keyword = parts[0].replace(" ","")

    if sentence == "什么是{}？".format(keyword):
        return True
    elif sentence == "如何{}？".format(keyword):
        return True
    else:
        return False

def write_json(results,output_path):
    # 写入新的JSON文件
    with open(output_path, 'w', encoding="utf-8") as file:
        json.dump(results, file,ensure_ascii=False,indent=4)