from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from tqdm import trange,tqdm
import json

chatglm_path = "chatglm3-6b/chatglm3-6b"

def get_context(model,question,data,key_vecs) -> str:
    """
    根据问题与索引标题的相似度进行检索，按照相似度从大到小返回索引序列
    """
    keylist = list(data.keys())
    result = model.encode(question, normalize_embeddings=True)
    scores = [ result @ vec.T for vec in key_vecs]

    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    return [keylist[i] for i in sorted_indices]

def get_answer(question,data,prompt_template,similarity_model,chatglm,key_vecs,
               top_k = 1,
               loop = True) -> str:
    """
    通过 句子相似度匹配模型 && ChatGLM 进行QA
    """
    keyword = get_context(similarity_model,question,data,key_vecs)
    info = ""
    for i in range(top_k):
        if keyword[i] in data:
            info += data[keyword[i]]

    inputs = {
        "question": question,
        "info":info
        }

    tokenizer = AutoTokenizer.from_pretrained(chatglm_path, trust_remote_code=True)
    # Execute the chain
    response, history = chatglm.chat(tokenizer,prompt_template[0].format(question,inputs["info"]), history=[])
    if loop:
        response, history = chatglm.chat(tokenizer,prompt_template[1].format(question), history=history)

    return response,keyword[:top_k],inputs["info"]

def run_query(prompt_template
              ,similarity_model
              ,chatglm
              ,data
              ,question_path = 'data/测试问题.json'
              ,test=False):
    ## 保存key对应的embedding
    key_vecs = []
    key_list = list(data.keys())
    for key in tqdm(key_list, desc="key vectors"):
        embedding = similarity_model.encode(key, normalize_embeddings=True)
        key_vecs.append(embedding)

    print("save key vectors done!!!")

    with open(question_path, 'r', encoding='utf-8') as f:
        question_list = json.load(f)
    questions = [q['question'] for q in question_list]

    answers = []

    if test == True:
        questions = questions[:5]
        
    for question in tqdm(questions,desc="question"):
        ret,keyword,related_str = get_answer(question,data,prompt_template,similarity_model,chatglm,key_vecs)
        sample = {"question": question, "keyword": keyword, "related_str": related_str, "answer": ret}
        answers.append(sample)
    
    if test == False:
        with open(f"result/answer.json", 'w', encoding='utf-8') as f:
            json.dump(answers, f, ensure_ascii=False, indent=4)
    else:
        with open(f"result/test.json", 'w', encoding='utf-8') as f:
            json.dump(answers, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    ## 句子相似度匹配模型
    device = 'cuda:0'
    sbert_id = './sbert/fengshan/stella-base-zh/'
    similarity_model = SentenceTransformer(sbert_id,device = device)

    ## ChatGLM 加载
    chatglm = AutoModel.from_pretrained(chatglm_path, trust_remote_code=True, device=device)
    chatglm = chatglm.eval()
    # 读取外部知识
    with open("header_all.json", "r", encoding="utf-8") as file:
        json_data = file.read()
    data = json.loads(json_data)

    prompt_template = ["""根据已知信息筛选出最相关内容并简洁地回答问题。问题是： {}已知信息有： {}""",
    """请简要地总结上述回答，只保留与问题最相关的部分。问题是：{}"""]

    # run_query(prompt_template,similarity_model,chatglm,data,'data/test.json')
    run_query(prompt_template,similarity_model,chatglm,data)
    pass