import os
import sys
import json


from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.docstore.document import Document
from langchain.llms import ChatGLM
from langchain.chains import LLMChain

from src.llm.template_manager import template_manager
from src.parse_pdf import parse_page_of_content
from src.utils import normalized_question, load_docs_from_jsonl, save_docs_to_jsonl, query_db



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



endpoint_url = ("http://127.0.0.1:29501")


def run_query(question_path = 'data/测试问题.json', test=True):
    with open(question_path, 'r', encoding='utf-8') as f:
        question_list = json.load(f)
    questions = [q['question'] for q in question_list]

    # load in embedding model
    model_name = "/home/lzw/.hf_models/stella-base-zh-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"} ,
        encode_kwargs={"normalize_embeddings": False})

    # construct a LLMchain
    config = {
        "temperature": 0.5,
        "top_p": 0.6,
        "max_new_tokens": 128,
    }

    llm = ChatGLM(
        endpoint_url=endpoint_url,
        history=[],
        **config,
        model_kwargs={"sample_model_args": False},
    )
    chain = LLMChain(llm=llm, prompt=template_manager.get_template(0))

    page_doc = load_docs_from_jsonl('data/page_documents.jsonl')
    page_id2doc = dict()
    
    # offset = 1
    for page in page_doc:
        page_id2doc[page.metadata["page"] + 1] = page

    index_db = FAISS.load_local('vector_store/all_index', embeddings)
    content_db = FAISS.load_local('vector_store/all_content', embeddings)

    answers = []

    if test == True:
        questions = questions[:5]
        
    for question in questions:
        # related_str, key_word = query_db(db, question, page_id2doc)
        question = normalized_question(question)
        related_str, key_word = query_db(index_db, content_db, question)
        result = chain(dict(question=question, related_str="<SEP>".join(related_str)))
        sample = {"question": question, "keyword": key_word, "related_str": related_str, "answer": result['text']}
        answers.append(sample)

    
    if test == False:
        with open(f"result/{config['temperature']}_top_p{config['top_p']}_answer.json", 'w', encoding='utf-8') as f:
            json.dump(answers, f, ensure_ascii=False, indent=4)
    else:
        with open(f"result/test.json", 'w', encoding='utf-8') as f:
            json.dump(answers, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    # construct_content_index()
    run_query(test=False)
    pass