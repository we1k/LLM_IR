import os
import sys

import json

from typing import Dict, Tuple, Iterable
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS

def construct_content_index():
    # # read documents
    # documents, table_of_content = parse_page_of_content()
    # save_docs_to_jsonl(documents, 'data/page_documents.jsonl')

    # # # page_index --> page_image, page_text, page_vector?
    # page_id2doc = dict()
    # for page in documents:
    #     page_id2doc[page.metadata["page"]] = page

    # all_sub_sections = {sub_k : sub_v for _, v in table_of_content.items() for sub_k, sub_v in v.items() }
    # print(all_sub_sections)

    # load in huggingface model
    model_name = "/home/lzw/.hf_models/stella-base-zh-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cuda"} ,
        encode_kwargs={"normalize_embeddings": False})

    # construct content index docs
    # content_index_docs = []
    # for sub_section_title, page_ids in all_sub_sections.items():
    #     content_index_docs.append(Document(page_content=sub_section_title, metadata={"page_ids": page_ids}))

    # db = FAISS.from_documents(content_index_docs, embeddings)
    # db.save_local("vector_store/page")

    content_documents = load_docs_from_jsonl("data/section_documents.jsonl")
    content_db = FAISS.from_documents(content_documents, embeddings)
    content_db.save_local("vector_store/content")


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


def query_page_db(db, question, page_id2doc):
    # 查询accuracy, 默认选取top_k = 1
    ret_docs = db.similarity_search(question, k=2)

    # 根据 title 进行段落切分
    key_word = ret_docs[0].page_content + "|" + ret_docs[1].page_content
    page_content = page_id2doc[ret_docs[0].metadata["page_ids"][0]].page_content + "<SEP>" + page_id2doc[ret_docs[1].metadata["page_ids"][0]].page_content
    
    print("page:", ret_docs[0].metadata["page_ids"][0], "\n\nkey_word:", key_word, "\n\ncontent", page_content)

    related_str = "|".join([key_word] + page_content.split(key_word)[1:])
    print("*" * 50)


    return related_str, key_word

def query_db(index_db, content_db, question, threshold=-120):
    # https://python.langchain.com/docs/integrations/vectorstores/faiss#similarity-search-with-filtering

    key_words = [(doc.page_content, score) for (doc, score) in index_db.similarity_search_with_relevance_scores(question, k=2) if score > threshold]
    
    related_str = []
    for key_word in key_words:
        ret_docs = content_db.similarity_search(question, filter={"key_word": key_word[0]}, k=1)
        for doc in ret_docs:
            related_str += [doc.page_content]

    return related_str, key_words, 


def query_sentence_db(db, question, neighbor=2):
    text_documents = load_docs_from_jsonl("data/texts.jsonl")
    ret = db.similarity_search_with_relevance_scores(question, k=1)
    id = ret[0][0].metadata["id"] - 1
    related_str = text_documents[id].page_content

    for i in range(neighbor + 1):
        related_str += "<SEP>" + "\n".join(text_documents[id+i+1].page_content.split("\n")  [1:])
    # prev_str, second_str = text_documents[id].page_content, text_documents[id+1].page_content
    # second_str = "\n".join(second_str.split("\n")[1:])
    # related_str = prev_str + "<SEP>" + second_str
    return related_str, "None"


SPECIAL_CASES_DICT = {
    "锁屏模式" : ["锁定状态"],
}
def normalized_question(question):
    ## TODO: parse the question, get rid of useless words
    # special case replacement
    
    for k, v in SPECIAL_CASES_DICT.items():
        for x in v:
            question = question.replace(x, k)
    return question