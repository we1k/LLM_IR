import os
import sys
import json

from argparse import ArgumentParser

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.docstore.document import Document
from langchain.llms import ChatGLM
from langchain.chains import LLMChain

from src.llm.template_manager import template_manager
from src.utils import normalized_question, load_docs_from_jsonl, save_docs_to_jsonl, query_db, query_sentence_db

endpoint_url = ("http://127.0.0.1:29501")


def run_query(question_path = 'data/测试问题.json', temperature=0.5, top_p=0.6, threshold=-80, test=True):
    if test == True:
        question_path = 'data/test.json'

    with open(question_path, 'r', encoding='utf-8') as f:
        question_list = json.load(f)
    questions = [q['question'] for q in question_list]

    # load in embedding model
    model_name = "/home/lzw/.hf_models/stella-base-zh-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu", } ,
        encode_kwargs={"normalize_embeddings": False})

    # construct a LLMchain
    config = {
        "temperature": temperature,
        "top_p": top_p,
        # "max_new_tokens": 48,
        "threshold" : threshold,
        "max_tokens": 48,
    }

    llm = ChatGLM(
        endpoint_url=endpoint_url,
        history=[],
        **config,
        model_kwargs={"sample_model_args": False},
    )
    # QA_chain = LLMChain(llm=llm, prompt=template_manager.get_fewshot_template(), verbose=True)
    QA_chain = LLMChain(llm=llm, prompt=template_manager.get_template(0), verbose=args.test==True)

    # Summarize_chain = LLMChain(llm=llm, prompt=template_manager.get_template(1))

    page_doc = load_docs_from_jsonl('data/page_documents.jsonl')
    page_id2doc = dict()
    
    # offset = 1
    for page in page_doc:
        page_id2doc[page.metadata["page"] + 1] = page

    index_db = FAISS.load_local('vector_store/all_index', embeddings)
    content_db = FAISS.load_local('vector_store/all_content', embeddings)
    sentence_db = FAISS.load_local('vector_store/sentence', embeddings)
    answers = []

        
    for question in questions:
        # related_str, key_word = query_db(db, question, page_id2doc)
        norm_question = normalized_question(question)
        related_str, key_word = query_db(index_db, content_db, norm_question)

        # key_words 检索到的关键词分数
        if len(key_word) >= 1 and key_word[0][1] > threshold:
            print("Using section db")
            result = QA_chain(dict(question=question, related_str=related_str[0]))
            # result = chain(dict(question=question, related_str="<SEP>".join(related_str)))
        else:
            print("Using sentence db")
            related_str, _ = query_sentence_db(sentence_db, question, k=2)
            result = QA_chain(dict(question=question, related_str="\n".join(related_str)))
            key_word += [("sentence", 0)]

        # final_result = Summarize_chain(dict(question=question, answer=result['text']))

        sample = {"question": question, "keyword": key_word, "related_str": related_str, "answer": result['text']}
        answers.append(sample)

    
    if test == False:
        with open(f"result/threshold{config['threshold']}_temperature{config['temperature']}_top_p{config['top_p']}_answer.json", 'w', encoding='utf-8') as f:
            json.dump(answers, f, ensure_ascii=False, indent=4)
    else:
        with open(f"result/test.json", 'w', encoding='utf-8') as f:
            json.dump(answers, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    # construct_content_index()
    parser = ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--threshold", default=-40, type=int)
    parser.add_argument("--temperature", default=0.5, type=float)
    parser.add_argument("--top_p", default=0.6, type=float)
    args = parser.parse_args()
    run_query(threshold=args.threshold, temperature=args.temperature, top_p=args.top_p, test=args.test)
    pass