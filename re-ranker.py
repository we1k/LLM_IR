import torch
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.utils import clean_related_str

model_path = "./rerank_model/bge-reranker-large"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path).cuda()
model.eval()


with open("result/related_str.json", 'r', encoding='UTF-8') as f:
    samples = json.load(f)

with torch.no_grad():
    for sample in samples:
        question = sample['question']
        related_str = sample['related_str'] # list

        pairs = [(question, r) for r in related_str]
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        # print(question, ":", scores)

        related_str_with_score = zip(related_str, scores)
        sort_related_str = sorted(related_str_with_score, key=lambda x: x[1], reverse=True)

        sample["related_str"] = clean_related_str([str for str, score in sort_related_str])[:5]

        # print("=====================================")

with open(f"result/related_str.json", 'w', encoding='utf-8') as f:
    json.dump(samples, f, ensure_ascii=False, indent=4)
