import os
import re
import json
from collections import defaultdict
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS, Chroma

from src.embeddings import BGEpeftEmbedding

DELIMITER = ['，', '。', '；', '–', '：', '！', '-', '、', '■', '□', '℃']

def save_docs_to_jsonl(array, file_path:str)->None:
    with open(file_path, 'w') as jsonl_file:
        for doc in array:
            jsonl_file.write(json.dumps(doc.dict(), ensure_ascii=False) + '\n')


def get_keywords():
    # File path to the outline document
    file_path = 'pdf_output/trainning_data.outline'

    # Read the entire file content
    with open(file_path, 'r') as file:
        file_content = file.read()

    # Use regular expression to find all instances of chapter names in <a> tags
    chapter_details = re.findall(
    r'<a class="l"[^>]*data-dest-detail=\'\[(\d+),[^\]]+\]\'>(.*?)\s*</a>',
    file_content
    )

    chapter_to_number_dict = {detail[1].strip(): int(detail[0]) for detail in chapter_details}
    chapter_names = [k.replace("&amp;", "&").strip() for k, v in chapter_to_number_dict.items()]
    return chapter_names


def build_sections(max_sentence_len=29):
    keywords = get_keywords()
    section_docs = []
    sections = defaultdict(str)
    chapter_name = ""
    tmp = ""

    with open("data/all.txt", 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            # 去除目录
            if ". . . ." in line or "目录" in line:
                continue
            if line.strip() in keywords:
                if chapter_name != "":
                    sections[chapter_name] += "<SEP>" + tmp
                    tmp = ""
                chapter_name = line.strip()
            else:
                tmp += line
    
    with open("data/section.json", 'w', encoding='UTF-8') as f:
        json.dump(sections, f, ensure_ascii=False, indent=4)

    sections[chapter_name] = tmp
    for chapter_name, text in sections.items():
        subsection_dict = {}
        text = text.replace("点击\n", "点击")
        text = text.replace("\n-", "-")
        text = text.replace("的\n", "的")
        sentences = text.split('\n')
        
        keyword = chapter_name
        cur_chunk = chapter_name + "\n"
        for sentence in sentences:
            sentence = sentence.strip("<SEP>")
            if len(sentence) == 0 or sentence.isdigit():
                continue
            # 大概率是目录
            # 可能包含章节数字 1.1 标题
            elif re.match(r"^\d*(\.\d+)?.*$", sentence) and not any(it in sentence for it in DELIMITER) and not sentence.startswith("0") and 1< len(sentence) <= max_sentence_len - 1:
                if cur_chunk.strip("\n") != keyword:
                    subsection_dict[keyword] = cur_chunk
                keyword = sentence
                cur_chunk = sentence + "\n"
            # 拼接上下句子
            elif len(sentence) >= max_sentence_len - 1 or ("，" in sentence and not sentence.endswith("。")):
                cur_chunk += sentence
            # 换行后第一个字符是分隔符
            elif any(sentence.startswith(it) for it in DELIMITER):
                cur_chunk = cur_chunk.strip("\n") + sentence + "\n"
            else:
                cur_chunk += sentence + "\n"

        # adding last chunk
        if cur_chunk.strip("\n") != keyword:
            subsection_dict[keyword] = cur_chunk

        for subkeyword, text_chunk in subsection_dict.items():
            if len(text_chunk.strip("<SEP>")) > 0 and not text_chunk.isalpha() > 0:
                # skip special char
                text_chunk = text_chunk.replace("<SEP>", "")
                section_docs.append(Document(page_content=text_chunk, metadata={"keyword": chapter_name, "subkeyword": subkeyword}))
        
    return section_docs

def preprocess(embedding_model, local_run=False, max_sentence_len=29):
    with open("data/raw.txt", 'r', encoding='UTF-8') as f:
        text = f.read()

    sections = re.split(r'!\[\]\(.+?\)', text)
    # 去掉页眉和页码
    for i in range(len(sections)):
        sections[i] = re.sub(rf'^.*?\n{i}\n', "", sections[i], flags=re.DOTALL)

    all_text = "".join(sections).replace("\n\n", "\n")
    with open("data/all.txt", 'w', encoding='UTF-8') as f:
        f.write(all_text)

    section_docs = build_sections(max_sentence_len)

    save_docs_to_jsonl(section_docs, "doc/section_docs.jsonl")

    all_keywords = [doc.metadata["keyword"] for doc in section_docs] + [doc.metadata["subkeyword"] for doc in section_docs]
    all_keywords = list(set(all_keywords))

    with open("data/keywords.txt", 'w', encoding='UTF-8') as f:
        f.write("\n".join(all_keywords))
        

    # get retriever !!
    # load in embedding model
    if "bge" in embedding_model:
        model_name = "/app/models/bge-large-zh-v1.5"
        embeddings = BGEpeftEmbedding(model_name)
    elif "stella" in embedding_model:
        if local_run:
            model_name = "/home/lzw/.hf_models/stella-base-zh-v2"
        else:
            model_name = "/app/models/stella-base-zh-v2"
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cuda"} ,
            encode_kwargs={"normalize_embeddings": False})
    elif "gte" in embedding_model:
        model_name = "/app/models/gte-large-zh"
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cuda"} ,
            encode_kwargs={"normalize_embeddings": False})
    

    db = FAISS.from_documents(section_docs, embeddings)
    db.save_local("vector_store/section_db")


    index_db = FAISS.from_texts(all_keywords, embeddings)
    index_db.save_local("vector_store/index_db")
    index_db = FAISS.load_local('vector_store/index_db', embeddings)

    # sentence cut
    chunk_size = 120
    chunk_overlap = 20
    sentence_splitter = RecursiveCharacterTextSplitter(
        separators=["。\n", "\n\n", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )

    sent_docs = sentence_splitter.split_documents(section_docs)
    
    # adding index and combine
    cur_doc_content = ""
    clean_sent_docs = []
    for doc in sent_docs:
        cur_doc_content += doc.page_content
        if len(doc.page_content) >= 50:
            doc.page_content = cur_doc_content
            doc.page_content = doc.page_content.replace("<SEP>", "")
            doc.page_content = doc.page_content.replace("■", "")
            doc.page_content = doc.page_content.replace("□", "")
            doc.page_content = doc.page_content.strip("。\n")
            doc.metadata['index'] = len(clean_sent_docs)
            clean_sent_docs.append(doc)
            cur_doc_content = ""
    sent_docs = clean_sent_docs


    save_docs_to_jsonl(sent_docs, "doc/sent_docs.jsonl")
    sent_db = FAISS.from_documents(sent_docs, embeddings)
    sent_db.save_local("vector_store/sentence_db")

if __name__ == '__main__':
    preprocess("stella")