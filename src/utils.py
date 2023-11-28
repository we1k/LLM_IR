import os
import sys
import json
import random
from typing import Dict, List, Tuple, Iterable, Optional

import torch
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from fuzzywuzzy import fuzz, process

SHORTEST_LENGTH = 1

def save_docs_to_jsonl(array:Iterable[Document], file_path:str)->None:
    with open(file_path, 'w') as jsonl_file:
        for doc in array:
            jsonl_file.write(json.dumps(doc.dict(), ensure_ascii=False) + '\n')

def load_docs_from_jsonl(file_path)->Iterable[Document]:
    array = []
    with open(file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = Document(**data)
            array.append(obj)
    return array


def query_db(index_db, content_db, question, threshold=-50):
    # https://python.langchain.com/docs/integrations/vectorstores/faiss#similarity-search-with-filtering

    key_words = [(doc.page_content, score) for (doc, score) in index_db.similarity_search_with_relevance_scores(question, k=2) if score > threshold]
    
    related_str = []
    for key_word in key_words:
        ret_docs = content_db.similarity_search(question, filter={"keyword": key_word}, k=1)
        for doc in ret_docs:
            related_str += [doc.page_content]

    return related_str, key_words


def query_sentence_db(retriever, question, neighbor=4):
    text_documents = load_docs_from_jsonl("data/texts.jsonl")
    related_sents = []

    ret_docs = retriever.get_relevant_documents(question)

    sent_set = set()
    for ret_doc in ret_docs:
        # if ret_doc.page_content in sent_set:
            # continue
        # sent_set.add(ret_doc.page_content)
        tmp_str = []
        id = ret_doc.metadata["id"]
        seen_sent = {}
        for i in range(neighbor):
            for sent in text_documents[id + i].page_content.split("\n"):
                if sent in seen_sent:
                    continue
                seen_sent[sent] = True
                tmp_str += [sent]
        related_sents += ["\n".join(tmp_str)]
    
    return related_sents


SPECIAL_CASES_DICT = {
    # "锁屏模式" : ["锁定状态"],
    # "颈椎撞击保护系统" : ["颈椎保护系统"],
}
def normalized_question(question):
    ## TODO: parse the question, get rid of useless words
    # special case replacement
    delimiters = ["如何", "？" , "什么", "哪些", "哪个", "哪种", "哪", "怎么", "怎样", "如果", ]
    for delimiter in delimiters:
        question = question.replace(delimiter, "")
    for k, v in SPECIAL_CASES_DICT.items():
        for x in v:
            question = question.replace(x, k)
    return question


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def adjusted_ratio_fn(str1, str2):
    if len(str1) + len(str2) == 0:
        return 0
    # 根据字符串长度计算一个权重，用于调整相似度得分
    len_weight = min(len(str1), len(str2)) / max(len(str1), len(str2), 1)
    
    # 计算标准的相似度得分
    similarity_score = fuzz.WRatio(str1, str2)
    
    # 根据长度权重调整得分
    score = similarity_score * len_weight
    return score

def clean_related_str(related_str, threshold=-80):
    """
        Input:
            related_str: List[str]
        Return:
            List[str]
    """
    if len(related_str) == 1:
        return related_str

    ## remove duplicate
    tmp_str = "<BOS>\n".join(related_str)
    parts = tmp_str.split("\n")
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
            if max_score > 75:
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


def remove_duplicate_strs(strings):
    if len(strings) == 1:
        return strings[0]

    unique_strings = [strings[0]]
    for string in strings[1:]:
        # 检查当前字符串是否与已有唯一字符串组中的任意一个相似
        _,max_score = process.extractOne(query=string, choices=unique_strings, scorer=adjusted_ratio_fn)
        if max_score > 75:
            continue
        else:
            unique_strings.append(string)

    ret_strs = clean_related_str(unique_strings)
    return "\n".join(ret_strs)

if __name__ == "__main__":
    strs = [
        '小计里程\n小计里程显示上次清零后车辆的行驶里程。小计里程常显于液晶左下角，与TRIP1界面中的小计里程为同一信息。激活方向盘按键复用后，在该界面下可以通过长按TRIP键或者长按方向盘的MODE按键单独对小计里程显示清零。\n仪表有自动记忆小计里程的功能。关闭启动开关，小计里程不会自动清零，但断开蓄电池后，小计里程将自动清零。\n小计里程界面显示，每0.1km更新一次。显示范围为0000.0~9999.9km，当达到最大值后，小计里程显示从0.0开始重新计算。',
        '小计里程\n小计里程显示上次清零后车辆的行驶里程。该界面下可以通过长按组合开关左侧TRI键，或者长按方向盘中的MODE按键（激活方向盘复用）单独对小计里程显示清零。仪表有自动记忆小计里程的功能。关闭启动开关，小计里程不会自动清零，但断开蓄电池后，小计里程将自动清零。\n小计里程界面显示，每0.1km更新一次。显示范围为0~9999.9km，当达到最大值后，小计里程显示从0.0开始重新计算。\n'
        ]
    print(remove_duplicate_strs(strs))