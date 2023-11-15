import os
import sys
import json
from tqdm import tqdm
import re
from argparse import ArgumentParser

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.docstore.document import Document
from langchain.llms import ChatGLM
from langchain.chains import LLMChain

import jieba
import jieba.analyse

from transformers import set_seed
from fuzzywuzzy import process

from src.preprocess import preprocess
from src.embeddings import BGEpeftEmbedding
from src.llm.template_manager import template_manager
from src.utils import normalized_question, adjusted_ratio_fn, load_docs_from_jsonl

set_seed(42)
endpoint_url = ("http://127.0.0.1:29501")

def run_query(args, question_path='data/测试问题.json'):
    if args.test == True:
        args.local_run = True
        question_path = 'data/test.json'

    with open(question_path, 'r', encoding='utf-8') as f:
        question_list = json.load(f)
    questions = [q['question'] for q in question_list]

    # load in embedding model
    if "bge" in args.embedding_model:
        model_name = "/home/lzw/.hf_models/bge-large-zh-v1.5"
        embeddings = BGEpeftEmbedding(model_name)
    elif "stella" in args.embedding_model:
        if "large" in args.embedding_model:
            model_name = "/home/lzw/.hf_models/stella-large-zh-v2"
        else:
            model_name = "/home/lzw/.hf_models/stella-base-zh-v2"
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cuda"} ,
            encode_kwargs={"normalize_embeddings": False})
    elif "gte" in args.embedding_model:
        model_name = "./model/gte-large-zh"
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cuda"} ,
            encode_kwargs={"normalize_embeddings": False})

    # construct a LLMchain

    if args.local_run:
        config = {
            "temperature": args.temperature,
            "top_p": args.top_p,
            # "max_new_tokens": 48,
            "threshold" : args.threshold,
            "max_tokens": 48,
        }

        llm = ChatGLM(
            endpoint_url=endpoint_url,
            history=[],
            **config,
            model_kwargs={"sample_model_args": False},
        )
        QA_chain = LLMChain(llm=llm, prompt=template_manager.get_template(0), verbose=args.test==True)

    # adding jieba dict
    jieba.load_userdict("data/keywords.txt")
    # jieba.analyse.set_idf_path("all.txt")

    # loading index db
    index_db = FAISS.load_local('vector_store/index_db', embeddings)
    content_db = FAISS.load_local('vector_store/section_db', embeddings)
    sentence_db = FAISS.load_local('vector_store/sentence_db', embeddings)
    sentence_retriever = sentence_db.as_retriever(search_kwargs={'k': 10})
    sent_reindex_retriever = sentence_db.as_retriever(search_kwargs={'k': 1})
    answers = []

    # load in keywords and abbr2word
    with open("data/keywords.txt", 'r', encoding='UTF-8') as f:
        all_keywords = f.read().split("\n")

    with open("data/abbr2word.json", 'r', encoding='UTF-8') as f:
        Abbr2word = json.load(f)

    with open("data/subkey2section_dict.json", 'r', encoding='UTF-8') as f:
        subkey2section_dict = json.load(f)
    
    subsection_keys = list(subkey2section_dict.keys())
    section_keys = list(set(subkey2section_dict.values()))

    # load in section and sentence docs
    section_docs = load_docs_from_jsonl("tmp/section_docs.jsonl")
    sent_docs = load_docs_from_jsonl("tmp/sent_docs.jsonl")

    id2sent_dict = {}
    for doc in sent_docs:
        id2sent_dict[doc.metadata['index']] = doc

    key2section_dict = {}
    for doc in section_docs:
        key2section_dict[doc.metadata['subkeyword']] = doc


    for question in tqdm(questions):
        # 替换缩写
        for k, v in Abbr2word.items():
            question = question.replace(k, v)

        # section db
        norm_question = normalized_question(question)

        # 通过关键词检索 section
        # jieba section cut
        keywords = []
        related_sections = []
        tags = jieba.analyse.extract_tags(norm_question, withWeight=False, allowPOS=())
        # print(tags)
        for tag in tags:
            if tag in all_keywords:
                keywords.append(tag)

        # TODO: 通过关键词检索 subsection
        # 通过关键词检索 subsection
        for keyword in keywords:
            if keyword in all_keywords:
                section_retriever = content_db.as_retriever(search_kwargs={'k': 2, "filter": {"keyword": keyword}})
                # related_sections += 
                
        # keywords embedding
        # for keyword in list(set([k[0] for k in keywords])):
        #     if keyword in section_keys:
        #         ret_docs = content_db.similarity_search(question, filter={"keyword": keyword}, k=2)
        #     elif keyword in subsection_keys:
        #         ret_docs = content_db.similarity_search(question, filter={"subkeyword": keyword}, k=2)
        #     for doc in ret_docs:
        #         related_sections += [doc.page_content]


        ### TODO:需要设置阈值
        # content_db.similarity_search_with
        section_retriever = content_db.as_retriever(search_kwargs={'k': 2})
        ret_docs = section_retriever.get_relevant_documents(question)
        related_sections += [doc.page_content for doc in ret_docs]

        # sentence db and fuzzywuzzy to rerank
        ret_docs = sentence_retriever.get_relevant_documents(question)
        related_sents = [doc.page_content for doc in ret_docs]

        # remove duplicate sent
        tmp_sents = []
        [tmp_sents.append(i) for i in related_sents if not i in tmp_sents]
        related_sents = tmp_sents
        
        # using fuzzywuzzy to rerank related str
        _related_sents = [str for str, score in process.extractBests(query=question, choices=related_sents, scorer=adjusted_ratio_fn)]

        print(_related_sents)
        related_sents = []
        for i in range(len(_related_sents)):
            concat_sent = ""
            ret_doc = sent_reindex_retriever.get_relevant_documents(_related_sents[i])[0]
            cur_idx = ret_doc.metadata["index"]
            pivot_keyword = ret_doc.metadata["subkeyword"]
            while True:
                if len(concat_sent) > 256 or id2sent_dict[cur_idx].metadata["subkeyword"] != pivot_keyword:
                    break
                concat_sent += id2sent_dict[cur_idx].page_content
                cur_idx += 1
            related_sents.append(concat_sent)


        # keywords 检索到的关键词分数
        if len(related_sections) >= 1:
            print(f"Using section db plus top {args.max_num_related_str-len(related_sections)} sentence db")
            related_sents = related_sents[:args.max_num_related_str-len(related_sections)]
        else:
            print("Using top args.max_num_related_str sentence db")
            related_sections = []
            related_sents = related_sents[:args.max_num_related_str]
            # 向后找完整个句子

            keywords += [("sentence", 0)]
        
        related_str = related_sections + related_sents

        if args.local_run:
            result = QA_chain(dict(question=question, related_str="\n".join(related_str)))
            sample = {"question": question, "keyword": keywords, "related_str": related_str, "answer": result['text']}
        else:
            sample = {"question": question, "keyword": keywords, "related_str": related_str}

        answers.append(sample)

    
    if args.test == False:
        with open(f"result/related_str.json", 'w', encoding='utf-8') as f:
            json.dump(answers, f, ensure_ascii=False, indent=4)
    else:
        with open(f"result/test.json", 'w', encoding='utf-8') as f:
            json.dump(answers, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    # construct_content_index()
    parser = ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--threshold", default=-120, type=int)
    parser.add_argument("--temperature", default=0.5, type=float)
    parser.add_argument("--top_p", default=0.6, type=float)
    parser.add_argument("--max_num_related_str", default=5, type=int)
    parser.add_argument("--local_run", action="store_true")
    parser.add_argument("--embedding_model", default="stella")
    args = parser.parse_args()
    # bge // stella // gte
    preprocess(args.embedding_model)
    run_query(args)
    # pass