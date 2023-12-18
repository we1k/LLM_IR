import os
import sys
import json
from tqdm import tqdm
import re
from argparse import ArgumentParser

from langchain.vectorstores.faiss import FAISS
from langchain.docstore.document import Document
from langchain.retrievers import BM25Retriever, EnsembleRetriever

import jieba
import jieba.analyse

from transformers import set_seed
from fuzzywuzzy import process

from src.preprocess import preprocess
from src.llm.template_manager import template_manager
from src.utils import normalized_question, load_docs_from_jsonl, load_embedding_model

set_seed(42)
# endpoint_url = ("http://127.0.0.1:29501")

def run_query(args, embeddings):
    if args.local_run == True:
        question_path = "data/all_question.json"
        # question_path = "data/test_question.json"
    else:
        question_path = "/tcdata/test_question.json"


    with open(question_path, 'r', encoding='utf-8') as f:
        question_list = json.load(f)
    questions = [q['question'] for q in question_list]

    # adding jieba dict
    jieba.load_userdict("data/keywords.txt")
    # jieba.analyse.set_idf_path("all.txt")

    # loading index db
    content_db = FAISS.load_local('vector_store/section_db', embeddings)
    sentence_db = FAISS.load_local('vector_store/sentence_db', embeddings)
    sentence_retriever = sentence_db.as_retriever(search_kwargs={'k': 15})
    answers = []

    # load in keywords and abbr2word
    # with open("data/keywords.txt", 'r', encoding='UTF-8') as f:
    #     all_keywords = f.read().split("\n")

    with open("data/abbr2word.json", 'r', encoding='UTF-8') as f:
        Abbr2word = json.load(f)

    # load in section and sentence docs
    section_docs = load_docs_from_jsonl("doc/section_docs.jsonl")
    sent_docs = load_docs_from_jsonl("doc/sent_docs.jsonl")

    id2sent_dict = {}
    for doc in sent_docs:
        id2sent_dict[doc.metadata['index']] = doc

    key2section_dict = {}
    for doc in section_docs:
        key2section_dict[doc.metadata['subkeyword']] = doc


    for question in tqdm(questions):
        # 替换缩写
        for k, v in Abbr2word.items():
            question = question.replace(k, v+f"({k})")
            if f"{v}({k})" not in question:
                question = question.replace(v, v+f"({k})")

        # section db
        norm_question = normalized_question(question)

        # 通过关键词检索 section
        keywords = []
        related_sections = []
        # tags = jieba.analyse.extract_tags(norm_question, withWeight=False, allowPOS=())
        # # print(tags)
        # for tag in tags:
        #     if tag in all_keywords:
        #         keywords.append(tag)

        # 通过关键词检索 subsection
        # for keyword in keywords:
        #     if keyword in all_keywords:
        #         section_retriever = content_db.as_retriever(search_kwargs={'k': 1, "filter": {"subkeyword": keyword}})
                

        # content_db.similarity_search_with
        ret_docs_with_score = content_db.similarity_search_with_relevance_scores(question, k=5)
        ret_docs = []
        for doc, score in ret_docs_with_score:
            # print(doc, score)
            if score > args.threshold:
                keywords.append((doc.metadata["subkeyword"], score))
                ret_docs.append(doc)
        related_sections += [doc.page_content for doc in ret_docs]

        # sentence db and fuzzywuzzy to rerank
        ret_docs = sentence_retriever.get_relevant_documents(question)
        related_sents = [doc.page_content for doc in ret_docs]


        # adding BM25 retrieval for sent_docs
        bm25_retriever = BM25Retriever.from_documents(sent_docs, k=10,
            preprocess_func=lambda x: list(jieba.cut_for_search(x))
        )

        bm25_ret_docs = bm25_retriever.get_relevant_documents(norm_question)
        related_sents += [doc.page_content for doc in bm25_ret_docs]
        # remove duplicate sent docs
        tmp_sents = []
        [tmp_sents.append(i) for i in related_sents if not i in tmp_sents]
        related_sents = tmp_sents
        
        related_str = related_sections + related_sents

        sample = {"question": question, "keyword": keywords, "related_str": related_str}

        answers.append(sample)

    
    with open(f"result/related_str.json", 'w', encoding='utf-8') as f:
        json.dump(answers, f, ensure_ascii=False, indent=4)
    # TODO:TEST
    # with open(f"result/unsort_related_str.json", 'w', encoding='utf-8') as f:
    #     json.dump(answers, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--threshold", default=0.3, type=float)
    parser.add_argument("--temperature", default=0.5, type=float)
    parser.add_argument("--top_p", default=0.6, type=float)
    parser.add_argument("--max_num_related_str", default=5, type=int)
    parser.add_argument("--max_sentence_len", default=19, type=int)
    parser.add_argument("--local_run", action="store_true")
    parser.add_argument("--embedding_model", default="stella")
    args = parser.parse_args()
    # bge // stella // gte
    embeddings = load_embedding_model(args.embedding_model, args.local_run) 
    preprocess(embeddings, args.max_sentence_len)
    run_query(args, embeddings)