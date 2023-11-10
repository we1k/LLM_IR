import os
import re
import json
from collections import defaultdict
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS, Chroma

MAX_KEYWORD_LEN = 20
MAX_SENTENCE_LEN = 30
DELIMITER = ['，', '。', '；', '–', '■', '□', '：', '！', '-', '的', '是']

def save_docs_to_jsonl(array, file_path:str)->None:
    with open(file_path, 'w') as jsonl_file:
        for doc in array:
            jsonl_file.write(json.dumps(doc.dict(), ensure_ascii=False) + '\n')


def get_keywords():
    # File path to the outline document
    file_path = 'out/QA.outline'

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


def build_sections():

    keywords = get_keywords()
    sections = defaultdict(str)
    chapter_name = ""
    tmp = ""

    with open("all.txt", 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            if line.strip() in keywords:
                if chapter_name != "":
                    sections[chapter_name] += "<SEP>" + tmp
                    tmp = ""
                chapter_name = line.strip()
            else:
                tmp += line
    sections[chapter_name] = tmp

    for chapter_name, text in sections.items():
        # 切分句子
        sentences = text.split('\n')
        # 处理每个句子
        processed_sentences = ""
        for sentence in sentences:
            if len(sentence) == 0:
                continue
            # 可能包含章节数字 1.1 标题
            elif re.match(r"^\d+(\.\d+){0,1,2}\s+.*$", sentence) and any(not sentence.endswith(it) for it in DELIMITER):
                processed_sentences += "\n" + sentence + "\n"
            # 句子长度小于最大关键词长度，且不包含分隔符
            elif len(sentence) < MAX_KEYWORD_LEN and not any(it in sentence for it in DELIMITER):
                processed_sentences += "\n" + sentence + "\n"
            # 拼接上下句子
            elif len(sentence) > MAX_SENTENCE_LEN - 2 or ("，" in sentence and not sentence.endswith("。")):
                processed_sentences += sentence
            # 换行后第一个字符是分隔符
            elif any(sentence.startswith(it) for it in DELIMITER):
                processed_sentences = processed_sentences.strip("\n") + sentence + "\n"
            else:
                processed_sentences += sentence + "\n"

        # 重新组合文本
        processed_sentences = processed_sentences.strip("\n")
        if len(processed_sentences) > 0:
            sections[chapter_name] = processed_sentences

    return sections

def main():
    keywords = get_keywords()
    # os.system("pdf2htmlEX --embed cfijo --dest-dir out data/QA.pdf")
    os.system("html2text out/QA.html utf-8 --ignore-links --escape-all > raw.txt")

    with open("raw.txt", 'r', encoding='UTF-8') as f:
        text = f.read()

    sections = re.split(r'!\[\]\(.+?\)', text)
    # 去掉页眉和页妈
    for i in range(len(sections)):
        sections[i] = re.sub(rf'^.*?\n{i}\n', "", sections[i], flags=re.DOTALL)

    all_text = "".join(sections).replace("\n\n", "\n")
    with open("all.txt", 'w', encoding='UTF-8') as f:
        f.write(all_text)

    sections = build_sections()
    chunk_size = 200
    chunk_overlap = 20


    # cut section
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        length_function=len,
        is_separator_regex=False
    )

    section_docs = text_splitter.create_documents(list(sections.values()), metadatas=[{"keyword": k} for k in sections.keys()])
    subsection_keys = []
    subkey2section_dict = defaultdict(str)
    # process section docs
    for doc in section_docs:
        sentences = doc.page_content.split("\n")
        # 第一句话可能是subsection，且第一句话长度小于最大关键词长度，且不包含分隔符
        if len(sentences) > 1 and len(sentences[0]) < MAX_KEYWORD_LEN:
            tmp = re.sub(r"^\d+(\.\d+){0,1,2}\s+.*$", "", sentences[0])
            tmp = tmp.replace("<SEP>", "")
            if not any(it in tmp for it in DELIMITER) and len(tmp) > 0:
                subsection_keys.append(tmp)
                doc.metadata['keyword'] = tmp
                subkey2section_dict[tmp] = doc.metadata['keyword']

    save_docs_to_jsonl(section_docs, "section_docs.jsonl")

    # save dict
    with open("subkey2section_dict.json", 'w', encoding='UTF-8') as f:
        json.dump(subkey2section_dict, f, ensure_ascii=False, indent=4)

    all_keywords = keywords + subsection_keys

    # get retriever !!
    model_name = "/home/lzw/.hf_models/stella-base-zh-v2"
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
    # adding index
    for i, doc in enumerate(sent_docs):
        doc.metadata["index"] = i

    save_docs_to_jsonl(sent_docs, "sent_docs.jsonl")
    sent_db = FAISS.from_documents(sent_docs, embeddings)
    sent_db.save_local("vector_store/sentence_db")

    # section_retriever = db.as_retriever(search_kwargs={'k': 5})
    # index_retriever = index_db.as_retriever(search_kwargs={'k': 2})

    # testing
    # query = "驾驶员状态监测系统是如何工作的？"
    # sent_retriever = sent_db.as_retriever(search_kwargs={'k': 5})
    # sent_retriever.get_relevant_documents(query)
    # index_retriever.get_relevant_documents(query)
    # section_retriever.get_relevant_documents(query)

if __name__ == '__main__':
    main()