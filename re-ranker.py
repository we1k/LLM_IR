import torch
import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.utils import clean_related_str

UPPER_BOUND = 10
LOWER_BOUND = -10
# setting hyperparams
BATCH_SIZE = 8

model_path = "./rerank_model/bge-reranker-large"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path).cuda()
model.eval()


with open("result/related_str.json", 'r', encoding='UTF-8') as f:
    samples = json.load(f)

with torch.no_grad():
    for i in range(0, len(samples), BATCH_SIZE):
        if i + BATCH_SIZE > len(samples):
            batch_samples = samples[i:]
        else:
            batch_samples = samples[i:i+BATCH_SIZE]
        
        batch_pairs = []
        for sample in batch_samples:
            question = sample['question']
            related_str = sample['related_str']  # list
            pairs = [(question, r) for r in related_str]
            batch_pairs.extend(pairs)

        # Tokenize and compute scores for the entire batch
        inputs = tokenizer(batch_pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        # Process scores for each sample in the batch
        score_index = 0
        for sample in batch_samples:
            related_str = sample['related_str']
            sample_scores = scores[score_index:score_index+len(related_str)]
            score_index += len(related_str)

            related_str_with_score = zip(related_str, sample_scores)
            sort_related_str_with_score = sorted(related_str_with_score, key=lambda x: x[1], reverse=True)[:5]
            
            while len(sort_related_str_with_score) > 3 and sort_related_str_with_score[-1][1] < 0:
                sort_related_str_with_score.pop()

            sample["no_answer"] = True if sort_related_str_with_score[0][1] < UPPER_BOUND else False
            sample["dont_answer"] = True if sort_related_str_with_score[0][1] < LOWER_BOUND else False
            sample["related_str"] = [str for str, score in sort_related_str_with_score]

with open(f"result/related_str.json", 'w', encoding='utf-8') as f:
    json.dump(samples, f, ensure_ascii=False, indent=4)
