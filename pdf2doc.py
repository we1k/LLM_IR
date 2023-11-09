import os
import re
import sys
import json

from bs4 import BeautifulSoup
from collections import defaultdict
from typing import Dict, Tuple, Iterable

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader, PyMuPDFLoader, PDFMinerPDFasHTMLLoader
from langchain.vectorstores.faiss import FAISS

from src.utils import save_docs_to_jsonl

def split_and_deduplicate(s):
    # 按照"。"为分隔符切分字符串
    parts = s.split("。")
    seen = set()
    deduplicated_parts = []
    # 去重，同时保证去重后的顺序不变
    for part in parts:
        if part not in seen:
            seen.add(part)
            deduplicated_parts.append(part)
    return "。".join(deduplicated_parts)


def parse_html(data):
    soup = BeautifulSoup(data.page_content,'html.parser')
    content = soup.find_all('div')
    # find all section names
    cur_fs = None
    cur_text = ''
    cur_page_id = 0
    snippets = []   # first collect all snippets that have the same font size
    for c in content:
        spans = c.find_all('span')
        # may contain multiple spans
        if len(spans) == None:
            continue
        for span in spans:
            st = span.get('style')
            if not st:
                continue
            fs = re.findall('font-size:(\d+)px',st)
            if not fs:
                continue
            fs = int(fs[0])
            if not cur_fs:
                cur_fs = fs
            if fs == cur_fs:
                cur_text += span.text
            else:
                cur_text = cur_text.strip()
                cur_text = re.sub(r'^\d+','',cur_text)
                cur_text = cur_text.replace('。\n','<SEP>').replace('\n', '').replace('<SEP>','。\n')
                if not re.match(r"^\d+$", cur_text) and len(cur_text) > 1:
                    if len(snippets) >= 1 and snippets[-1]['font_size'] == cur_fs:
                        snippets[-1]['sentence'] += cur_text
                    else:
                        snippets.append({"sentence":cur_text,"font_size":cur_fs})
                cur_fs = fs
                cur_text = span.text
    snippets.append({"sentence":cur_text,"font_size":cur_fs})

    # fs -> [len(all_text), len(|all_text|), set(all_text) )]
    fs_distribution_dict = defaultdict(lambda : [0, 0, set()])
    for s in snippets:
        s['sentence'] = s['sentence'].strip().replace("\n", "").replace(" ","").replace(".", "")
        fs_distribution_dict[s['font_size']][0] += len(s['sentence'])
        fs_distribution_dict[s['font_size']][1] += 1
        fs_distribution_dict[s['font_size']][2] |= set(s['sentence'])

    # for k, v in fs_distribution_dict.items():
    #     print(f"font_size: {k}, len(all_text): {v[0]}, len(|all_text|): {v[1]}, len(set(all_text)): {len(v[2])}")
    
    with open("data/sentence.json", "w", encoding='utf-8') as f:
        json.dump(snippets, f, ensure_ascii=False, indent=4)

    # processing font size to guess/accquire the section cut
    page_num = 354
    fs_list = []
    sorted_fs_len = sorted(fs_distribution_dict.items(), key=lambda x: x[1][0], reverse=True)
    text_fs = sorted_fs_len[0][0]
    for k, v in sorted_fs_len:
        # 删除字数过少的 font size
        # 删除平均字数过少的 font size
        # 删除|all_text|过低的 font size
        if k >= text_fs and v[0] > page_num * 1.5 and v[0] / v[1] > 4.1 and len(v[2]) > 50 :
            fs_list.append((k, v[0]/v[1], v[1],  len(v[2])))

    assert len(fs_list) >= 2, "not enough font size to pack for `TEXT` and `SECTION`"

    # unpack font size
    section_fs = -1
    for fs_info in fs_list:
        # 必须比正文字体大
        if fs_info[0] < text_fs:
            continue
        # 标题不可能太长 20 个字 
        if fs_info[1] < 20:
            section_fs = fs_info[0]
            break
    
    print(text_fs, section_fs)

    # Cut section
    section = {}
    cur_sec = None
    cur_text = ""
    for s in snippets:
        if s['font_size'] == section_fs:
            if cur_sec != None:
                section[cur_sec] = cur_text
            cur_sec = s['sentence']
            cur_text = ""
        else:
            cur_text += s['sentence']
        
    section[cur_sec] = cur_text
    setion_keys = section.keys()
    html_section_documents = convert_dict_to_doc(section, "data/html_section_documents.jsonl")
    return html_section_documents, setion_keys
    

def convert_dict_to_doc(dict, saved_doc_path):
    documents = []
    for k, v in dict.items():
        dict[k] = split_and_deduplicate(v)
        documents.append(Document(page_content=dict[k], metadata={"key_word":k}))
    save_docs_to_jsonl(documents, saved_doc_path)
    return documents


# def merge_documents(documents_list: Iterable[Document]) -> Document:
#     clean_documents_dict = defaultdict(str)
#     for documents in documents_list:
#         for doc in documents:
#             clean_documents_dict[doc.metadata['key_word']] = split_and_deduplicate(doc.page_content + clean_documents_dict[doc.metadata['key_word']])

#     merged_documents = [Document(page_content=v, metadata={"key_word":k}) for k, v in clean_documents_dict.items()]

#     return merged_documents


def main(pdf_path="data/QA.pdf"):
    # html cut section
    loader = PDFMinerPDFasHTMLLoader(pdf_path)
    data = loader.load()[0]
    html_section_documents, section_keys = parse_html(data)


    # text cut section
    loader = PyPDFLoader(pdf_path)
    pdf_docs = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
    )
    page_documents = text_splitter.split_documents(pdf_docs)

    # may not work
    all_text = ""
    for doc in page_documents:
        # 页码删除 and 页眉删除
        content = doc.page_content

        new_lines = []
        tmp = ""
        ## 修正换行
        for line in content.split("\n"):
            line = line.strip()
            tmp += line
            if line.endswith("。"):
                new_lines.append(tmp)
                tmp = ""
            elif line in section_keys:
                # 添加 <sub_section> 标签
                line = "<\sub_section>\n\n<sub_section>" + line
                new_lines.append(line)
                tmp = ""

        new_lines.append(tmp)
        content = "\n".join(new_lines)

        # all_text += f"\n<PAGE_SEP> page_id:{page_id}\n" + content
        all_text += content

    # skip table of content
    all_text = "\n".join(all_text.split("\n"))

    with open("data/test_all_text.txt", "w") as f:
        f.write(all_text)


    # make text splitter documents
    chunk_size = 100
    chunk_overlap = 50

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )

    sentence_docments = text_splitter.create_documents([all_text.replace("<\sub_section>\n\n<sub_section>", "")])

    save_docs_to_jsonl(sentence_docments, "data/sentence_documents.jsonl")

    # sub_section_documents
    sub_sections = all_text.split("<\sub_section>")
    sub_sections_dict = defaultdict(str)
    for sub_section in sub_sections:
        sub_section = sub_section.split("<sub_section>")[-1]
        keyword = sub_section.split("\n")[0]
        sub_sections_dict[keyword] += "\n".join(sub_section.split("\n")[1:])

    txt_section_documents = convert_dict_to_doc(sub_sections_dict, "data/txt_section_documents.jsonl")

    all_documents = html_section_documents + txt_section_documents
    save_docs_to_jsonl(all_documents, "data/test_all_documents.jsonl")


    # construct vector_store
    # model_name = "/home/lzw/.hf_models/stella-base-zh-v2"
    # embeddings = HuggingFaceEmbeddings(
    #     model_name=model_name,
    #     model_kwargs={"device": "cuda"} ,
    #     encode_kwargs={"normalize_embeddings": False})

    # all_index = [Document(page_content=doc.metadata["key_word"]) for doc in all_documents]
    # all_index_db = FAISS.from_documents(all_index, embeddings)
    # all_index_db.save_local("vector_store/all_index")

    # all_content = [Document(page_content=doc.page_content, metadata={"key_word":doc.metadata["key_word"]}) for doc in all_documents]
    # all_content_db = FAISS.from_documents(all_content, embeddings)
    # all_content_db.save_local("vector_store/all_content")

if __name__ == '__main__':
    # main()
    
    from src.utils import load_docs_from_jsonl
    test_all_docs = load_docs_from_jsonl("data/html_section_documents.jsonl")
    all_docs = load_docs_from_jsonl("data/txt_section_documents.jsonl")
    test_all_docs_keys = [doc.metadata['key_word'] for doc in test_all_docs]
    all_docs_keys = [doc.metadata['key_word'] for doc in all_docs]
    # print(set(all_docs_keys) - set(test_all_docs_keys))
    