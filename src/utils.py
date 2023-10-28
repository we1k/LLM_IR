import os
import sys

import json

from typing import Dict, Tuple, Iterable
from langchain.docstore.document import Document


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


def normalized_question(question):
    ## TODO: parse the question, get rid of useless words
    return question